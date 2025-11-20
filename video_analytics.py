#!/usr/bin/env python3
"""Покадровая аналитика внимания, текста и логотипов для коротких видео."""
from __future__ import annotations

import faulthandler

faulthandler.enable()

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.functional as F
from PIL import Image
from tqdm import tqdm

import boto3
from botocore.client import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# NOTE: we rely on the api package being importable from the repo root.
from api.app.models.artifact import Artifact  # type: ignore
from api.app.models.job import Job, JobStatus  # type: ignore
from api.app.models.video import Video, VideoStatus  # type: ignore

# Ensure torch.meshgrid uses explicit indexing to avoid deprecation warnings.
if not getattr(torch, "_meshgrid_default_indexing_set", False):
    _original_meshgrid = torch.meshgrid

    def _meshgrid_with_default_indexing(*tensors, **kwargs):
        if "indexing" not in kwargs:
            kwargs["indexing"] = "ij"
        return _original_meshgrid(*tensors, **kwargs)

    torch.meshgrid = _meshgrid_with_default_indexing  # type: ignore[assignment]
    F.meshgrid = _meshgrid_with_default_indexing  # type: ignore[assignment]
    setattr(torch, "_meshgrid_default_indexing_set", True)


# --- мини-замена torchvision.transforms -------------------
class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, copy=True)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        # HWC -> CHW, float32 [0,1]
        arr = torch.from_numpy(arr.transpose(2, 0, 1)).float().div(255.0)
        return arr


class Normalize:
    def __init__(self, mean, std):
        # mean, std: последовательности длины C
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class _T:
    Compose = Compose
    ToTensor = ToTensor
    Normalize = Normalize


T = _T()
# --- конец мини-замены ------------------------------------


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


def parse_minio_path(value: str) -> tuple[str, str]:
    """Extract bucket/key from an s3/http(s) URL that points to MinIO."""

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "s3"}:
        path = parsed.path.lstrip("/")
        if not path or "/" not in path:
            raise ValueError("URL must contain bucket and key, e.g. http://host/bucket/key")
        bucket, key = path.split("/", 1)
        return bucket, key

    raise ValueError("Unsupported URL: provide http(s) or s3 URL pointing to MinIO")


def build_minio_client(endpoint: str | None, access_key: str | None, secret_key: str | None):
    if not endpoint or not access_key or not secret_key:
        raise ValueError("MINIO endpoint/access/secret are required for remote operations")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )


def download_minio_object(client, uri: str, destination_dir: Path) -> Path:
    bucket, key = parse_minio_path(uri)
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_path = destination_dir / Path(key).name
    client.download_file(bucket, key, str(local_path))
    return local_path


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
    transform_image: Compose
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


def build_db_sessionmaker(database_url: str | None):
    if not database_url:
        return None
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


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
    transform: Compose,
    target_size: int,
    device: torch.device,
) -> torch.Tensor:
    from os2d.os2d.utils import get_image_size_after_resize_preserving_aspect_ratio  # type: ignore

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


def update_statuses(
    db,
    *,
    job,
    video,
    job_status: JobStatus | None = None,
    video_status: VideoStatus | None = None,
    progress: int | None = None,
    error: str | None = None,
):
    if not db:
        return

    changed = False
    if job is not None and job_status is not None:
        job.status = job_status
        changed = True
    if job is not None and progress is not None:
        job.progress = progress
        changed = True
    if job is not None and error is not None:
        job.error = error
        changed = True
    if video is not None and video_status is not None:
        video.status = video_status
        changed = True
    if changed:
        db.commit()


def register_artifact(db, *, video_id: str | None, bucket: str, key: str, artifact_type: str):
    if db is None or video_id is None:
        return
    uri = f"s3://{bucket}/{key}"
    artifact = Artifact(video_id=video_id, type=artifact_type, uri=uri)
    db.add(artifact)
    db.commit()


async def process_video(
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

    frames_info = await asyncio.to_thread(read_video_frames, video_path, seconds, frame_skip)
    height, width = frames_info.frame_size

    centerbias_template = np.load(DEEPGAZE_DIR / "centerbias_mit1003.npy")
    centerbias_log = build_centerbias(height, width, uniform=False, template=centerbias_template)
    centerbias_tensor = torch.from_numpy(centerbias_log).float().to(device)

    logo_templates = await asyncio.to_thread(load_logo_templates_from_path, logo_path, os2d_ctx)

    per_frame: list[dict[str, Any]] = []
    volatility_pairs: list[dict[str, Any]] = []
    chunk = max(1, chunk_size)
    diagonal = float(np.hypot(height, width))
    if diagonal <= 0:
        diagonal = 1.0

    prev_prob_map: np.ndarray | None = None
    prev_max_position: tuple[int, int] | None = None
    prev_frame_index: int | None = None
    prev_timestamp: float | None = None
    total_frames = len(frames_info.frames_bgr)
    progress = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    for start in range(0, total_frames, chunk):
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
            text_boxes = await asyncio.to_thread(
                detect_text_boxes,
                frame_bgr,
                craft_model,
                device,
                TEXT_THRESHOLD,
                LINK_THRESHOLD,
                LOW_TEXT,
                CANVAS_SIZE,
                MAG_RATIO,
            )
            logo_detections = await asyncio.to_thread(
                detect_logos_on_frame,
                frame_bgr,
                logo_templates,
                os2d_ctx,
            )

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

            if prev_prob_map is not None and prev_max_position is not None and prev_frame_index is not None:
                diff_map = prob_map - prev_prob_map
                vol1_mean = float(np.mean(np.abs(diff_map)))
                vol1_rms = float(np.sqrt(np.mean(np.square(diff_map))))
                vol1_max = float(np.max(np.abs(diff_map)))

                prev_pos = np.array(prev_max_position, dtype=np.float32)
                curr_pos = np.array(max_position, dtype=np.float32)
                dist_px = float(np.linalg.norm(curr_pos - prev_pos))
                dist_norm = float(dist_px / diagonal)

                volatility_pairs.append(
                    {
                        "frame_a": prev_frame_index,
                        "frame_b": frame_index,
                        "time_a": prev_timestamp,
                        "time_b": timestamp,
                        "vol1_mean": vol1_mean,
                        "vol1_rms": vol1_rms,
                        "vol1_max": vol1_max,
                        "vol2_value": dist_px,
                        "vol2_normalized": dist_norm,
                    }
                )

            prev_prob_map = prob_map
            prev_max_position = max_position
            prev_frame_index = frame_index
            prev_timestamp = timestamp
        progress.update(len(batch_frames))

    progress.close()

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

    def build_volatility_series(field: str) -> list[dict[str, Any]]:
        return [
            {
                "frame_a": pair["frame_a"],
                "frame_b": pair["frame_b"],
                "time_a": pair["time_a"],
                "time_b": pair["time_b"],
                "value": pair[field],
            }
            for pair in volatility_pairs
        ]

    series["volatility1_mean"] = build_volatility_series("vol1_mean")
    series["volatility1_rms"] = build_volatility_series("vol1_rms")
    series["volatility1_max"] = build_volatility_series("vol1_max")
    series["volatility2"] = [
        {
            "frame_a": pair["frame_a"],
            "frame_b": pair["frame_b"],
            "time_a": pair["time_a"],
            "time_b": pair["time_b"],
            "value": pair["vol2_value"],
            "normalized": pair["vol2_normalized"],
        }
        for pair in volatility_pairs
    ]

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


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Получение аналитики по видео")
    parser.add_argument("--video", type=Path, default=None, help="Путь к видео")
    parser.add_argument("--logo", type=Path, default=None, help="Путь к логотипу")
    parser.add_argument("--video-url", type=str, default=None, help="MinIO HTTP/S3 URL к видео")
    parser.add_argument("--logo-url", type=str, default=None, help="MinIO HTTP/S3 URL к логотипу")
    parser.add_argument("--job-id", type=str, default=None, help="ID job для обновления статуса")
    parser.add_argument("--video-id", type=str, default=None, help="ID видео для обновления статуса/артефакта")
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="Postgres DATABASE_URL",
    )
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default=os.getenv("MINIO_ENDPOINT_INTERNAL") or os.getenv("MINIO_ENDPOINT_PUBLIC"),
        help="MinIO endpoint (внутренний)",
    )
    parser.add_argument(
        "--minio-access-key",
        type=str,
        default=os.getenv("MINIO_ACCESS_KEY"),
        help="MinIO access key",
    )
    parser.add_argument(
        "--minio-secret-key",
        type=str,
        default=os.getenv("MINIO_SECRET_KEY"),
        help="MinIO secret key",
    )
    parser.add_argument(
        "--results-bucket",
        type=str,
        default=os.getenv("MINIO_BUCKET_RESULTS"),
        help="Bucket для сохранения analytics.json",
    )
    parser.add_argument(
        "--result-key",
        type=str,
        default=None,
        help="Ключ внутри results bucket (по умолчанию video_id/video_analytics.json)",
    )
    parser.add_argument(
        "--skip-minio-upload",
        action="store_true",
        help="Не загружать результат в MinIO",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analytics.json"),
        help="Путь к JSON-отчёту",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Длительность анализируемого фрагмента (по умолчанию весь ролик)",
    )
    parser.add_argument("--frame-skip", type=int, default=1, help="Использовать каждый N-й кадр")
    parser.add_argument("--chunk-size", type=int, default=8, help="Размер батча для инференса")
    parser.add_argument("--device", type=str, default=None, help="Устройство (cpu/cuda)")
    args = parser.parse_args()

    if args.video_url is None:
        if args.video is None:
            raise ValueError("Укажите --video или --video-url")
        if not args.video.exists():
            raise FileNotFoundError(f"Видео не найдено: {args.video}")

    if args.logo_url is None and args.logo is not None and not args.logo.exists():
        raise FileNotFoundError(f"Логотип не найден: {args.logo}")

    minio_client = None
    need_minio = bool(args.video_url or args.logo_url or not args.skip_minio_upload)
    if need_minio:
        minio_client = build_minio_client(
            args.minio_endpoint, args.minio_access_key, args.minio_secret_key
        )
    SessionLocal = build_db_sessionmaker(args.database_url)
    db = SessionLocal() if SessionLocal and (args.job_id or args.video_id) else None
    job = db.get(Job, args.job_id) if db and args.job_id else None
    video = db.get(Video, args.video_id) if db and args.video_id else None

    update_statuses(
        db,
        job=job,
        video=video,
        job_status=JobStatus.STARTED,
        video_status=VideoStatus.PROCESSING,
        progress=0,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = args.video
            logo_path = args.logo

            if args.video_url:
                if not minio_client:
                    raise ValueError("MinIO клиент не инициализирован для загрузки видео")
                video_path = download_minio_object(minio_client, args.video_url, tmp_path)

            if args.logo_url:
                if not minio_client:
                    raise ValueError("MinIO клиент не инициализирован для загрузки логотипа")
                logo_path = download_minio_object(minio_client, args.logo_url, tmp_path)

            device = torch.device(
                args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            deepgaze_model, craft_model, os2d_ctx = load_models(device)
            await process_video(
                video_path=video_path,
                logo_path=logo_path,
                deepgaze_model=deepgaze_model,
                craft_model=craft_model,
                os2d_ctx=os2d_ctx,
                seconds=args.seconds,
                frame_skip=args.frame_skip,
                chunk_size=args.chunk_size,
                device=device,
                output_path=args.output,
            )

        if not args.skip_minio_upload:
            if not minio_client:
                raise ValueError("MinIO клиент обязателен для загрузки результатов")
            if not args.results_bucket:
                raise ValueError("Укажите --results-bucket или MINIO_BUCKET_RESULTS")
            default_key = (
                f"{args.video_id}/video_analytics.json"
                if args.video_id
                else args.output.name
            )
            result_key = args.result_key or default_key
            minio_client.upload_file(
                str(args.output.resolve()), args.results_bucket, result_key
            )
            register_artifact(
                db,
                video_id=args.video_id,
                bucket=args.results_bucket,
                key=result_key,
                artifact_type="video_analytics.json",
            )

        update_statuses(
            db,
            job=job,
            video=video,
            job_status=JobStatus.DONE,
            video_status=VideoStatus.DONE,
            progress=100,
        )
    except Exception as exc:
        update_statuses(
            db,
            job=job,
            video=video,
            job_status=JobStatus.FAILED,
            video_status=VideoStatus.FAILED,
            error=str(exc),
        )
        raise
    finally:
        if db is not None:
            db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
