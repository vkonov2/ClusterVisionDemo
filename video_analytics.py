#!/usr/bin/env python3
"""Покадровая аналитика внимания, текста и логотипов для коротких видео."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEEPGAZE_DIR = MODEL_DIR / "deepgaze_pytorch"
CRAFT_DIR = MODEL_DIR / "craft"
OS2D_DIR = MODEL_DIR / "os2d"
sys.path.insert(0, str(MODEL_DIR))

from deepgaze_pytorch.deepgaze2e import DeepGazeIIE  # type: ignore  # noqa: E402
from craft.craft import CRAFT  # type: ignore  # noqa: E402
from craft import craft_utils  # type: ignore  # noqa: E402
from os2d.os2d.config import cfg  # type: ignore  # noqa: E402
from os2d.os2d.modeling.model import build_os2d_from_config  # type: ignore  # noqa: E402
from os2d.os2d.structures.feature_map import FeatureMapSize  # type: ignore  # noqa: E402

TEXT_THRESHOLD = 0.7
LINK_THRESHOLD = 0.4
LOW_TEXT = 0.4
CANVAS_SIZE = 1280
MAG_RATIO = 1.5
LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LOGO_SCORE_THRESHOLD = 0.45


@dataclass
class VideoFrames:
    frames_bgr: List[np.ndarray]
    fps: float

    @property
    def frame_size(self) -> Tuple[int, int]:
        if not self.frames_bgr:
            raise ValueError("No frames captured from video")
        h, w = self.frames_bgr[0].shape[:2]
        return h, w


@dataclass
class Os2dModelContext:
    """Хранит подготовленные объекты модели OS2D."""

    net: Any
    box_coder: Any
    transform_image: T.Compose
    device: torch.device
    frame_target_size: int
    class_image_size: int
    nms_iou_threshold: float
    nms_score_threshold: float
    max_detections: int
    feature_map_size_cls: Any


@dataclass
class LogoTemplate:
    name: str
    tensor: torch.Tensor
    path: Path


@dataclass
class LogoDetection:
    template: LogoTemplate
    score: float
    box: tuple[int, int, int, int]


def load_models(device: torch.device) -> tuple[DeepGazeIIE, CRAFT, Os2dModelContext]:
    # DeepGaze
    deepgaze_model = DeepGazeIIE(pretrained=False)
    deepgaze_model.to(device)
    deepgaze_model.eval()

    # CRAFT
    craft_model = CRAFT()
    state_dict = torch.load(CRAFT_DIR / "craft_mlt_25k.pth", map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    craft_model.load_state_dict(state_dict)
    craft_model.to(device)
    craft_model.eval()

    # OS2D
    cfg.is_cuda = device.type == "cuda"
    cfg.init.model = str(OS2D_DIR / "os2d_v2-train.pth")
    cfg.train.do_training = False
    cfg.eval.scales_of_image_pyramid = [1.0]
    cfg.eval.nms_iou_threshold = 0.3
    cfg.eval.nms_score_threshold = float("-inf")
    cfg.output.path = ""
    cfg.visualization.eval.max_detections = 30
    cfg.visualization.eval.score_threshold = float("-inf")

    net, box_coder, _, img_normalization, _ = build_os2d_from_config(cfg)
    net.eval()
    net.to(device)

    transform_image = T.Compose(
        [T.ToTensor(), T.Normalize(img_normalization["mean"], img_normalization["std"])]
    )

    frame_target_size = 800
    nms_iou_threshold = cfg.eval.nms_iou_threshold
    nms_score_threshold = cfg.eval.nms_score_threshold
    max_detections = cfg.visualization.eval.max_detections

    os2d_ctx = Os2dModelContext(
        net=net,
        box_coder=box_coder,
        transform_image=transform_image,
        device=device,
        frame_target_size=frame_target_size,
        class_image_size=int(cfg.model.class_image_size),
        nms_iou_threshold=nms_iou_threshold,
        nms_score_threshold=nms_score_threshold,
        max_detections=max_detections,
        feature_map_size_cls=FeatureMapSize,
    )

    return deepgaze_model, craft_model, os2d_ctx


def read_video_frames(video_path: Path | None, seconds: float | None, frame_skip: int) -> VideoFrames:
    """Read video frames up to the requested duration."""

    if video_path is None:
        raise ValueError("Video path must be provided")

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames
    if seconds is not None and fps > 0:
        max_frames = min(total_frames, int(fps * seconds))

    collected: List[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break
        if frame_index % frame_skip == 0:
            collected.append(frame_bgr)
        frame_index += 1
        if frame_index >= max_frames:
            break

    cap.release()
    if not collected:
        raise RuntimeError("No frames were read from the video")

    return VideoFrames(frames_bgr=collected, fps=fps)


def logsumexp_np(values: np.ndarray) -> float:
    max_val = float(values.max())
    stable = values - max_val
    return max_val + np.log(np.exp(stable).sum())


def build_centerbias(height: int, width: int, *, uniform: bool, template: np.ndarray | None) -> np.ndarray:
    if uniform or template is None:
        return np.zeros((height, width), dtype=np.float32)
    resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_NEAREST)
    resized = resized.astype(np.float32)
    resized -= logsumexp_np(resized)
    return resized


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1)))
    return tensor.float()


def log_density_to_prob(log_density: np.ndarray) -> np.ndarray:
    stable = log_density - log_density.max()
    prob = np.exp(stable)
    total = prob.sum()
    if total <= 0:
        return np.zeros_like(prob)
    return prob / total


def normalize_mean_variance(
    img: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    result = img.astype(np.float32)
    result -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    result /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return result


def resize_aspect_ratio(
    img: np.ndarray,
    square_size: int,
    interpolation: int = cv2.INTER_LINEAR,
    mag_ratio: float = 1.5,
) -> tuple[np.ndarray, float]:
    height, width, _ = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = float(square_size)

    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    target_h = max(1, target_h)
    target_w = max(1, target_w)

    resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    padded = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    padded[:target_h, :target_w] = resized
    return padded, 1 / ratio if ratio > 0 else 1.0


def detect_text_boxes(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    canvas_size: int,
    mag_ratio: float,
) -> list[np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized, target_ratio = resize_aspect_ratio(frame_rgb, canvas_size, mag_ratio=mag_ratio)
    ratio = 1 / target_ratio if target_ratio > 0 else 1.0

    norm = normalize_mean_variance(resized)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _ = model(tensor)

    score_text = y[0, :, :, 0].detach().cpu().numpy()
    score_link = y[0, :, :, 1].detach().cpu().numpy()

    boxes, _ = craft_utils.getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly=False,
    )

    if not boxes:
        return []

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio, ratio)
    return [np.array(box, dtype=np.float32) for box in boxes]


def preprocess_logo_image(
    image: Image.Image,
    transform: T.Compose,
    target_size: int,
    device: torch.device,
) -> torch.Tensor:
    from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio  # type: ignore

    height, width = image.size[1], image.size[0]
    new_height, new_width = get_image_size_after_resize_preserving_aspect_ratio(
        h=height,
        w=width,
        target_size=target_size,
    )
    if (new_width, new_height) != image.size:
        image = image.resize((new_width, new_height))
    tensor = transform(image).to(device)
    return tensor


def load_logo_templates_from_path(
    logo_path: Path | None,
    context: Os2dModelContext,
) -> list[LogoTemplate]:
    if logo_path is None:
        return []

    if logo_path.is_dir():
        raise ValueError("Logo path must be an image file, not a directory.")
    if logo_path.suffix.lower() not in LOGO_EXTENSIONS:
        raise ValueError("Unsupported logo format. Provide an image file (png/jpg/etc.)")

    with Image.open(logo_path) as img:
        tensor = preprocess_logo_image(
            img.convert("RGB"),
            context.transform_image,
            context.class_image_size,
            context.device,
        )
    return [LogoTemplate(name=logo_path.stem, tensor=tensor, path=logo_path)]


def detect_logos_on_frame(
    frame_bgr: np.ndarray,
    templates: list[LogoTemplate],
    context: Os2dModelContext,
) -> list[LogoDetection]:
    if not templates:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    frame_tensor = preprocess_logo_image(
        image,
        context.transform_image,
        context.frame_target_size,
        context.device,
    )
    input_batch = frame_tensor.unsqueeze(0)

    with torch.no_grad():
        loc_pred, class_pred, _, _, transform_corners = context.net(
            images=input_batch,
            class_images=[template.tensor for template in templates],
        )

    feature_map_size = context.feature_map_size_cls(img=input_batch)
    boxes = context.box_coder.decode_pyramid(
        [loc_pred[0]],
        [class_pred[0]],
        [feature_map_size],
        list(range(len(templates))),
        nms_score_threshold=context.nms_score_threshold,
        nms_iou_threshold=context.nms_iou_threshold,
        transform_corners_pyramid=[transform_corners[0]],
    )

    if boxes.bbox_xyxy.numel() == 0:
        return []

    scores = boxes.get_field("scores").detach().cpu().numpy()
    labels = boxes.get_field("labels").detach().cpu().numpy().astype(int)
    coords = boxes.bbox_xyxy.detach().cpu().numpy()

    input_height = int(input_batch.size(-2))
    input_width = int(input_batch.size(-1))
    frame_height, frame_width = frame_bgr.shape[:2]

    scale_x = frame_width / max(1, input_width)
    scale_y = frame_height / max(1, input_height)

    detections: list[LogoDetection] = []
    for score, label, (x1, y1, x2, y2) in zip(scores, labels, coords):
        if score < LOGO_SCORE_THRESHOLD or label >= len(templates):
            continue

        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))

        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            LogoDetection(
                template=templates[label],
                score=float(score),
                box=(x1, y1, x2, y2),
            )
        )

    detections.sort(key=lambda item: item.score, reverse=True)
    if len(detections) > context.max_detections:
        detections = detections[: context.max_detections]
    return detections


def compute_concentration(prob_map: np.ndarray, top_fraction: float) -> float:
    flat = prob_map.flatten()
    total = float(flat.sum())
    if total <= 0:
        return 0.0
    count = max(1, int(np.ceil(flat.size * top_fraction)))
    top_indices = np.argpartition(flat, -count)[-count:]
    top_sum = float(flat[top_indices].sum())
    return top_sum / total


def compute_text_energy_ratio(prob_map: np.ndarray, boxes: Sequence[np.ndarray]) -> float:
    if not boxes:
        return 0.0

    mask = np.zeros(prob_map.shape, dtype=np.float32)
    for box in boxes:
        contour = np.round(box).astype(np.int32)
        cv2.fillPoly(mask, [contour], 1.0)

    energy_inside = float((prob_map * mask).sum())
    mean_value = float(prob_map.mean())
    if mean_value <= 1e-12:
        return 0.0
    pixels_in_boxes = float(mask.sum())
    if pixels_in_boxes <= 0:
        return 0.0
    expected_energy = mean_value * pixels_in_boxes
    if expected_energy <= 1e-12:
        return 0.0
    return energy_inside / expected_energy


def compute_logo_energy_ratio(prob_map: np.ndarray, detections: Sequence[LogoDetection]) -> float:
    if not detections:
        return 0.0

    mask = np.zeros(prob_map.shape, dtype=np.float32)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1.0, thickness=-1)

    energy_inside = float((prob_map * mask).sum())
    mean_value = float(prob_map.mean())
    if mean_value <= 1e-12:
        return 0.0
    pixels_in_boxes = float(mask.sum())
    if pixels_in_boxes <= 0:
        return 0.0
    expected_energy = mean_value * pixels_in_boxes
    if expected_energy <= 1e-12:
        return 0.0
    return energy_inside / expected_energy


def serialize_boxes(boxes: Sequence[np.ndarray]) -> list[list[list[float]]]:
    return [box.astype(float).tolist() for box in boxes]


def serialize_logo_detections(detections: Sequence[LogoDetection]) -> list[dict[str, Any]]:
    return [
        {
            "name": detection.template.name,
            "score": detection.score,
            "box": list(map(int, detection.box)),
        }
        for detection in detections
    ]


def process_video(
    video_path: Path | None,
    logo_path: Path | None,
    deepgaze_model: DeepGazeIIE,
    craft_model: CRAFT,
    os2d_ctx: Os2dModelContext,
    seconds: float | None,
    frame_skip: int,
    chunk_size: int,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    """Processes a video, runs all detectors per frame, and exports analytics to JSON."""

    frames_info = read_video_frames(video_path, seconds, frame_skip)
    height, width = frames_info.frame_size

    centerbias_template = np.load(DEEPGAZE_DIR / "centerbias_mit1003.npy")
    centerbias_log = build_centerbias(height, width, uniform=False, template=centerbias_template)
    centerbias_tensor = torch.from_numpy(centerbias_log).float().to(device)

    logo_templates = load_logo_templates_from_path(logo_path, os2d_ctx)

    per_frame: list[dict[str, Any]] = []
    chunk = max(1, chunk_size)

    for start in range(0, len(frames_info.frames_bgr), chunk):
        batch_frames = frames_info.frames_bgr[start : start + chunk]
        tensors = [frame_to_tensor(frame) for frame in batch_frames]
        images = torch.stack(tensors).to(device)
        cb = centerbias_tensor.unsqueeze(0).expand(images.size(0), -1, -1)

        with torch.no_grad():
            log_density = deepgaze_model(images, cb)
        prob_maps = log_density.squeeze(1).detach().cpu().numpy()

        for offset, frame_bgr in enumerate(batch_frames):
            frame_index = start + offset
            prob_map = log_density_to_prob(prob_maps[offset])
            text_boxes = detect_text_boxes(
                frame_bgr,
                craft_model,
                device,
                TEXT_THRESHOLD,
                LINK_THRESHOLD,
                LOW_TEXT,
                CANVAS_SIZE,
                MAG_RATIO,
            )
            logo_detections = detect_logos_on_frame(frame_bgr, logo_templates, os2d_ctx)

            mean_value = float(prob_map.mean())
            peak_value = float(prob_map.max())
            conc10 = compute_concentration(prob_map, 0.10)
            conc20 = compute_concentration(prob_map, 0.20)
            text_ratio = compute_text_energy_ratio(prob_map, text_boxes)
            logo_ratio = compute_logo_energy_ratio(prob_map, logo_detections)
            max_position = np.unravel_index(int(prob_map.argmax()), prob_map.shape)
            timestamp = (frame_index / frames_info.fps) if frames_info.fps > 0 else None

            per_frame.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": timestamp,
                    "mean": mean_value,
                    "peak": peak_value,
                    "concentration_10": conc10,
                    "concentration_20": conc20,
                    "text_energy_ratio": text_ratio,
                    "logo_energy_ratio": logo_ratio,
                    "max_position": [int(max_position[1]), int(max_position[0])],
                    "text_boxes": serialize_boxes(text_boxes),
                    "logos": serialize_logo_detections(logo_detections),
                }
            )

    metric_keys = [
        "mean",
        "peak",
        "concentration_10",
        "concentration_20",
        "text_energy_ratio",
        "logo_energy_ratio",
    ]
    series = {
        key: [
            {
                "frame": item["frame_index"],
                "time": item["time_seconds"],
                "value": item[key],
            }
            for item in per_frame
        ]
        for key in metric_keys
    }

    analytics = {
        "video": str(video_path) if video_path is not None else None,
        "fps": frames_info.fps,
        "seconds_limit": seconds,
        "frame_skip": frame_skip,
        "frames_processed": len(per_frame),
        "logo_templates": [template.name for template in logo_templates],
        "series": series,
        "frames": per_frame,
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Analytics saved to {output_path}")
    return analytics


def main() -> int:
    parser = argparse.ArgumentParser(description="Получение аналитики по видео")
    parser.add_argument("--video", type=Path, default=None, help="Путь к видео")
    parser.add_argument("--logo", type=Path, default=None, help="Путь к логотипу")
    parser.add_argument("--output", type=Path, default=Path("analytics.json"), help="Путь к JSON-отчёту")
    parser.add_argument("--seconds", type=float, default=None, help="Длительность анализируемого фрагмента (по умолчанию весь ролик)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Использовать каждый N-й кадр")
    parser.add_argument("--chunk-size", type=int, default=8, help="Размер батча для инференса")
    parser.add_argument("--device", type=str, default=None, help="Устройство (cpu/cuda)")
    args = parser.parse_args()

    if args.video is not None and not args.video.exists():
        raise FileNotFoundError(f"Видео не найдено: {args.video}")
    if args.logo is not None and not args.logo.exists():
        raise FileNotFoundError(f"Логотип не найден: {args.logo}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    deepgaze_model, craft_model, os2d_ctx = load_models(device)
    process_video(
        video_path=args.video,
        logo_path=args.logo,
        deepgaze_model=deepgaze_model,
        craft_model=craft_model,
        os2d_ctx=os2d_ctx,
        seconds=args.seconds,
        frame_skip=args.frame_skip,
        chunk_size=args.chunk_size,
        device=device,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
