"""Подсветка логотипов на видео с помощью модели OS2D."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm


warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
)


CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "os2d"
REPO_DIR = CACHE_DIR / "repo"
REPO_URL = "https://github.com/aosokin/os2d.git"
WEIGHTS_DIR = CACHE_DIR / "models"
DEFAULT_MODEL = "os2d_v2-train.pth"
WEIGHTS_IDS: dict[str, str] = {
    "os2d_v2-train.pth": "1l_aanrxHj14d_QkCpein8wFmainNAzo8",
    "os2d_v1-train.pth": "1ByDRHMt1x5Ghvy7YTYmQjmus9bQkvJ8g",
    "os2d_v2-init.pth": "1sr9UX45kiEcmBeKHdlX7rZTSA4Mgt0A7",
}

DEFAULT_INPUT_DIR = Path("data/videos")
DEFAULT_LOGO_DIR = Path("data/logos")
DEFAULT_OUTPUT_DIR = Path("outputs/os2d")
DEFAULT_INTERMEDIATE_SUFFIX = "-os2d-silent.mp4"
DEFAULT_OUTPUT_SUFFIX = "-os2d.mp4"

LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mpg", ".mpeg"}


@dataclass
class LogoTemplate:
    """Данные о логотипе, используемые для детекции."""

    name: str
    path: Path
    tensor: torch.Tensor
    color: tuple[int, int, int]

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class LogoDetection:
    """Результат обнаружения логотипа на кадре."""

    template: LogoTemplate
    score: float
    box: tuple[int, int, int, int]


@dataclass
class VideoJob:
    """Описывает задачу обработки одного видео."""

    video_path: Path
    output_video: Path
    intermediate_video: Path
    templates: list[LogoTemplate]
    seconds: float | None
    score_threshold: float


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


def ensure_os2d_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Клонирует репозиторий OS2D при необходимости."""

    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Клонирую OS2D в {repo_dir}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def download_file_from_google_drive(file_id: str, destination: Path, chunk_size: int = 32768) -> None:
    """Скачивает файл из Google Drive без сторонних зависимостей."""

    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=30)
    response.raise_for_status()

    token = _extract_confirm_token(response)
    if token is not None:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

    _save_response_content(response, destination, chunk_size)


def _extract_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response: requests.Response, destination: Path, chunk_size: int) -> None:
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total is not None else None
    temp_path = destination.with_suffix(destination.suffix + ".part")

    downloaded = 0
    with temp_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size):
            if not chunk:
                continue
            file_obj.write(chunk)
            downloaded += len(chunk)

    temp_path.rename(destination)

    if total_bytes is not None and downloaded != total_bytes:
        raise RuntimeError(
            "Размер скачанного файла не совпадает с ожидаемым объёмом из заголовка"
        )


def ensure_os2d_weights(model_name: str = DEFAULT_MODEL) -> Path:
    """Гарантирует наличие весов OS2D для выбранной модели."""

    candidate = Path(model_name)
    if candidate.exists():
        return candidate.resolve()

    weights_id = WEIGHTS_IDS.get(model_name)
    if weights_id is None:
        raise FileNotFoundError(
            f"Неизвестны веса модели {model_name}. Укажите путь к локальному файлу или одно из имён: {sorted(WEIGHTS_IDS)}"
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = (WEIGHTS_DIR / model_name).resolve()
    if destination.exists():
        return destination

    print(f"Скачиваю веса модели {model_name}...")
    download_file_from_google_drive(weights_id, destination)
    if not destination.exists():
        raise RuntimeError(f"Не удалось скачать веса модели {model_name}")
    return destination


def _color_from_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return (int(b / 2 + 96), int(g / 2 + 96), int(r / 2 + 96))


def load_os2d_model(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
    frame_target_size: int = 1500,
    nms_iou_threshold: float = 0.3,
    nms_score_threshold: float = float("-inf"),
    max_detections: int = 30,
) -> Os2dModelContext:
    """Загружает модель OS2D и возвращает контекст для инференса."""

    repo_dir = ensure_os2d_repo()
    weights_path = ensure_os2d_weights(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from os2d.config import cfg  # type: ignore
    from os2d.modeling.model import build_os2d_from_config  # type: ignore
    from os2d.engine.optimization import create_optimizer  # type: ignore
    from os2d.utils import (  # type: ignore
        get_trainable_parameters,
        mkdir,
        set_random_seed,
        setup_logger,
    )
    from os2d.structures.feature_map import FeatureMapSize  # type: ignore

    cfg.is_cuda = device.type == "cuda"
    cfg.init.model = str(weights_path)
    cfg.train.do_training = False
    cfg.eval.scales_of_image_pyramid = [1.0]
    cfg.eval.nms_iou_threshold = nms_iou_threshold
    cfg.eval.nms_score_threshold = nms_score_threshold
    cfg.output.path = ""
    cfg.visualization.eval.max_detections = max_detections
    cfg.visualization.eval.score_threshold = float("-inf")

    mkdir(CACHE_DIR / "logs")
    setup_logger("OS2D", None)
    set_random_seed(cfg.random_seed, cfg.is_cuda)

    net, box_coder, _, img_normalization, optimizer_state = build_os2d_from_config(cfg)

    parameters = get_trainable_parameters(net)
    create_optimizer(parameters, cfg.train.optim, optimizer_state)

    net.eval()
    if cfg.is_cuda:
        net.cuda()
    else:
        net.cpu()

    transform_image = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(img_normalization["mean"], img_normalization["std"]),
        ]
    )

    return Os2dModelContext(
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


def preprocess_image(
    image: Image.Image,
    transform: T.Compose,
    target_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Изменяет размер изображения с сохранением пропорций и нормализацией."""

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


def load_logo_templates(
    video_path: Path,
    logos_root: Path,
    context: Os2dModelContext,
) -> list[LogoTemplate]:
    """Загружает изображения логотипов для видео и подготавливает тензоры."""

    logos_root = logos_root.expanduser()
    stem = video_path.stem
    candidate_dir = logos_root / stem

    logo_paths: list[Path] = []
    if candidate_dir.is_dir():
        for path in sorted(candidate_dir.iterdir()):
            if path.suffix.lower() in LOGO_EXTENSIONS:
                logo_paths.append(path)
    else:
        for ext in LOGO_EXTENSIONS:
            candidate = logos_root / f"{stem}{ext}"
            if candidate.exists():
                logo_paths.append(candidate)

    templates: list[LogoTemplate] = []
    for logo_path in sorted(logo_paths):
        with Image.open(logo_path) as img:
            rgb = img.convert("RGB")
            tensor = preprocess_image(
                rgb,
                context.transform_image,
                context.class_image_size,
                context.device,
            )
        templates.append(
            LogoTemplate(
                name=logo_path.stem,
                path=logo_path,
                tensor=tensor,
                color=_color_from_name(logo_path.stem),
            )
        )

    return templates


def detect_logos_on_frame(
    frame_bgr: np.ndarray,
    templates: list[LogoTemplate],
    context: Os2dModelContext,
    score_threshold: float,
) -> list[LogoDetection]:
    """Возвращает список обнаруженных логотипов на кадре."""

    if not templates:
        return []

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    frame_tensor = preprocess_image(
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
    frame_height, frame_width = frame_bgr.shape[0], frame_bgr.shape[1]

    scale_x = frame_width / max(1, input_width)
    scale_y = frame_height / max(1, input_height)

    detections: list[LogoDetection] = []
    for score, label, (x1, y1, x2, y2) in zip(scores, labels, coords):
        if score < score_threshold or label >= len(templates):
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


def draw_detections(frame_bgr: np.ndarray, detections: Iterable[LogoDetection]) -> np.ndarray:
    """Рисует обнаруженные логотипы на кадре."""

    if not detections:
        return frame_bgr

    result = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = detection.template.color
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.template.display_name}: {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        text_x = x1
        text_y = max(y1 - 4, text_height + 2)
        cv2.rectangle(
            result,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            result,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    return result


def mux_audio(
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    reencode: bool = True,
) -> None:
    """Объединяет подсвеченное видео с оригинальной аудиодорожкой."""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
    ]

    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    else:
        cmd += ["-c:v", "copy"]

    cmd += ["-c:a", "copy", "-shortest", str(output_video)]

    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            "Не удалось объединить видео и аудио.\n"
            f"STDOUT: {result.stdout.decode('utf-8', errors='ignore')}\n"
            f"STDERR: {result.stderr.decode('utf-8', errors='ignore')}"
        )


def process_video(job: VideoJob, context: Os2dModelContext, keep_intermediate: bool) -> None:
    """Запускает обработку одного видео."""

    video_path = job.video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    intermediate_video = job.intermediate_video.resolve()
    intermediate_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать видеофайл: {intermediate_video}")

    max_frames = None
    if job.seconds is not None and job.seconds > 0:
        max_frames = int(min(total_frames or int(job.seconds * fps), job.seconds * fps))

    progress_total = max_frames if max_frames is not None else total_frames or None
    progress = tqdm(total=progress_total, desc=video_path.name, unit="кадр")

    processed = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            detections = detect_logos_on_frame(
                frame,
                templates=job.templates,
                context=context,
                score_threshold=job.score_threshold,
            )
            highlighted = draw_detections(frame, detections)
            writer.write(highlighted)

            processed += 1
            progress.update(1)

            if max_frames is not None and processed >= max_frames:
                break
    finally:
        progress.close()
        cap.release()
        writer.release()

    mux_audio(intermediate_video, video_path, job.output_video.resolve())

    if not keep_intermediate and intermediate_video.exists():
        intermediate_video.unlink()


def find_videos(input_dir: Path) -> list[Path]:
    """Возвращает список видео в указанной директории."""

    videos: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    return videos


def build_jobs(
    args: argparse.Namespace,
    context: Os2dModelContext,
) -> list[VideoJob]:
    """Формирует список задач на обработку."""

    videos: list[Path] = []
    if args.video is not None:
        video_path = args.video.expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Указанное видео не найдено: {video_path}")
        videos = [video_path]
    else:
        input_dir = args.input_dir.expanduser()
        if not input_dir.exists():
            raise FileNotFoundError(f"Каталог с видео не найден: {input_dir}")
        videos = find_videos(input_dir)

    jobs: list[VideoJob] = []
    for video_path in videos:
        templates = load_logo_templates(video_path, args.logos_dir, context)
        if not templates:
            print(f"Пропускаю {video_path.name}: не найдены логотипы в {args.logos_dir}")
            continue

        output_dir = args.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        intermediate = output_dir / f"{video_path.stem}{DEFAULT_INTERMEDIATE_SUFFIX}"
        output_video = output_dir / f"{video_path.stem}{DEFAULT_OUTPUT_SUFFIX}"

        jobs.append(
            VideoJob(
                video_path=video_path,
                output_video=output_video,
                intermediate_video=intermediate,
                templates=templates,
                seconds=args.seconds,
                score_threshold=args.score_threshold,
            )
        )

    if not jobs:
        raise RuntimeError("Не найдено ни одного видео для обработки")

    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подсветка логотипов на видео с помощью OS2D",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Каталог с видеофайлами",
    )
    parser.add_argument(
        "--logos-dir",
        type=Path,
        default=DEFAULT_LOGO_DIR,
        help="Каталог с изображениями логотипов",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для сохранения результатов",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Обработать только указанный видеофайл",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Имя предобученной модели OS2D или путь к локальному файлу весов",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для инференса (например, cuda или cpu)",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=1500,
        help="Целевая длина большей стороны кадра перед подачей в модель",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.45,
        help="Порог уверенности для отображения детекций",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.3,
        help="IoU-порог для нефункционирующего подавления неверных дубликатов (NMS)",
    )
    parser.add_argument(
        "--nms-score",
        type=float,
        default=float("-inf"),
        help="Порог уверенности перед NMS",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=30,
        help="Максимальное число детекций на кадр",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Ограничить обработку первыми N секундами",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Не удалять временное видео без аудио",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context = load_os2d_model(
        model_name=args.model,
        device=device,
        frame_target_size=args.frame_size,
        nms_iou_threshold=args.nms_iou,
        nms_score_threshold=args.nms_score,
        max_detections=args.max_detections,
    )

    jobs = build_jobs(args, context)
    for job in jobs:
        process_video(job, context, keep_intermediate=args.keep_intermediate)
        print(f"Готово: {job.output_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
