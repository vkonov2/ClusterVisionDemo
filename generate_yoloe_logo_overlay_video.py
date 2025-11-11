"""Выделение логотипов брендов на видео с помощью модели YOLOE и шаблонов."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm


DEFAULT_INPUT_DIR = Path("data/videos")
DEFAULT_LOGO_DIR = Path("data/logos")
DEFAULT_OUTPUT_DIR = Path("outputs/logos")
DEFAULT_MODEL = "yoloe-11l-seg.pt"
DEFAULT_INTERMEDIATE_SUFFIX = "-logos-silent.mp4"
DEFAULT_OUTPUT_SUFFIX = "-logos.mp4"

LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

YOLOE_WEIGHT_URLS: dict[str, str] = {
    "yoloe-11l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt",
    "yoloe-v8l-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt",
    "yoloe-11l-seg-pf.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg-pf.pt",
    "yoloe-v8l-seg-pf.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg-pf.pt",
}

TEMPLATE_SCALES: tuple[float, ...] = (0.45, 0.55, 0.65, 0.8, 0.9, 1.0, 1.15, 1.3, 1.45, 1.6)
TEMPLATE_ANGLES: tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
NMS_IOU_THRESHOLD = 0.35
MAX_TEMPLATE_MATCHES_PER_LOGO = 8


@dataclass(frozen=True)
class LogoTemplate:
    """Хранит предобработанные данные для логотипа."""

    name: str
    image_bgr: np.ndarray
    image_gray: np.ndarray
    histogram: np.ndarray
    width: int
    height: int
    color: tuple[int, int, int]

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


@dataclass(frozen=True)
class LogoMatch:
    """Совпадение логотипа на кадре."""

    logo: LogoTemplate
    box: tuple[int, int, int, int]
    score: float
    method: str
    confidence: float | None = None
    scale: float | None = None
    angle: float | None = None


@dataclass(frozen=True)
class VideoJob:
    """Описание задачи обработки одного видео."""

    video_path: Path
    logos: Sequence[LogoTemplate]
    output_video: Path
    intermediate_video: Path
    seconds: float | None
    confidence_threshold: float
    histogram_threshold: float
    template_threshold: float
    keep_intermediate: bool
    merge_audio: bool


YOLOE_CACHE_DIR = Path(".cache/yoloe")
TEMPLATE_VARIANT_CACHE: dict[str, list[tuple[np.ndarray, float, float]]] = {}


def ensure_ultralytics() -> None:
    """Гарантирует наличие пакета ultralytics в окружении."""

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("Устанавливаю пакет ultralytics для модели YOLOE...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ultralytics>=8.2.0"],
            check=True,
        )
        import ultralytics  # type: ignore  # noqa: F401


def ensure_weight_file(model_name: str) -> Path:
    """Возвращает путь к файлу весов YOLOE, скачивая его при необходимости."""

    candidate = Path(model_name)
    if candidate.exists():
        return candidate

    url = YOLOE_WEIGHT_URLS.get(model_name)
    if url is None:
        raise FileNotFoundError(
            f"Файл весов {model_name} не найден и не известен среди заранее определённых ссылок"
        )

    YOLOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = YOLOE_CACHE_DIR / model_name
    if destination.exists():
        return destination

    print(f"Скачиваю веса модели {model_name}...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as target:
            shutil.copyfileobj(response.raw, target)

    return destination


def load_yoloe_model(model_name: str = DEFAULT_MODEL, device: torch.device | None = None):
    """Загружает модель YOLOE из пакета ultralytics."""

    ensure_ultralytics()
    from ultralytics import YOLO  # type: ignore

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_path = ensure_weight_file(model_name)

    model = YOLO(str(weight_path))
    model.to(device)
    return model, device


def _color_from_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return (int(b / 2 + 96), int(g / 2 + 96), int(r / 2 + 96))


def _compute_histogram(image_bgr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _rotate_template(image_gray: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает шаблон на заданный угол, расширяя холст при необходимости."""

    if abs(angle) < 1e-3:
        return image_gray

    height, width = image_gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image_gray,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _get_template_variants(logo: LogoTemplate) -> list[tuple[np.ndarray, float, float]]:
    """Возвращает масштабированные и повернутые варианты шаблона."""

    cache_key = logo.name
    if cache_key in TEMPLATE_VARIANT_CACHE:
        return TEMPLATE_VARIANT_CACHE[cache_key]

    variants: list[tuple[np.ndarray, float, float]] = []
    for scale in TEMPLATE_SCALES:
        scaled_width = int(round(logo.width * scale))
        scaled_height = int(round(logo.height * scale))
        if scaled_width < 4 or scaled_height < 4:
            continue
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
        scaled = cv2.resize(
            logo.image_gray,
            (scaled_width, scaled_height),
            interpolation=interpolation,
        )
        for angle in TEMPLATE_ANGLES:
            rotated = _rotate_template(scaled, angle)
            variants.append((rotated, scale, angle))

    TEMPLATE_VARIANT_CACHE[cache_key] = variants
    return variants


def _prepare_logo_templates(image_paths: Sequence[Path]) -> list[LogoTemplate]:
    templates: list[LogoTemplate] = []
    for image_path in image_paths:
        if image_path.suffix.lower() not in LOGO_EXTENSIONS:
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hist = _compute_histogram(image_bgr)
        height, width = image_bgr.shape[:2]
        templates.append(
            LogoTemplate(
                name=image_path.stem,
                image_bgr=image_bgr,
                image_gray=image_gray,
                histogram=hist,
                width=width,
                height=height,
                color=_color_from_name(image_path.stem),
            )
        )

    if not templates:
        raise RuntimeError("Не удалось подготовить изображения логотипов для сравнения")
    return templates


def load_logo_templates(logo_source: Path) -> list[LogoTemplate]:
    """Загружает изображения логотипов и подготавливает их к сравнению."""

    logo_source = logo_source.expanduser()
    if logo_source.is_dir():
        image_paths = sorted(logo_source.iterdir())
        return _prepare_logo_templates(image_paths)
    if logo_source.is_file():
        return _prepare_logo_templates([logo_source])

    raise FileNotFoundError(f"Каталог или файл с логотипами не найден: {logo_source}")


def detect_with_yolo(
    frame_bgr: np.ndarray,
    model,
    confidence_threshold: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Возвращает детекции YOLOE в формате (bbox, confidence)."""

    results = model.predict(frame_bgr, verbose=False, conf=confidence_threshold)
    detections: list[tuple[tuple[int, int, int, int], float]] = []
    if not results:
        return detections

    height, width = frame_bgr.shape[:2]
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for coords, confidence in zip(xyxy, confs):
            x1, y1, x2, y2 = coords.astype(int)
            x1 = int(np.clip(x1, 0, width - 1))
            x2 = int(np.clip(x2, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            y2 = int(np.clip(y2, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(((x1, y1, x2, y2), float(confidence)))
    return detections


def _histogram_similarity(patch_bgr: np.ndarray, logo: LogoTemplate) -> float:
    resized = cv2.resize(patch_bgr, (logo.width, logo.height), interpolation=cv2.INTER_AREA)
    hist = _compute_histogram(resized)
    return float(cv2.compareHist(hist, logo.histogram, cv2.HISTCMP_CORREL))


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Вычисляет IoU двух прямоугольников."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union = max(area_a + area_b - intersection, 1e-6)
    return intersection / union


def _template_match_multi(
    frame_gray: np.ndarray,
    logo: LogoTemplate,
    template_threshold: float,
    existing_boxes: list[tuple[int, int, int, int]],
) -> list[LogoMatch]:
    """Ищет несколько совпадений шаблона с учётом масштабов и поворотов."""

    fh, fw = frame_gray.shape[:2]
    raw_candidates: list[tuple[tuple[int, int, int, int], float, float, float]] = []

    for template, scale, angle in _get_template_variants(logo):
        th, tw = template.shape[:2]
        if th > fh or tw > fw:
            continue
        result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        if result.size == 0:
            continue
        locations = np.argwhere(result >= template_threshold)
        for y, x in locations:
            score = float(result[y, x])
            box = (int(x), int(y), int(x + tw), int(y + th))
            raw_candidates.append((box, score, scale, angle))

    raw_candidates.sort(key=lambda item: item[1], reverse=True)

    selected: list[LogoMatch] = []
    occupied = existing_boxes[:]
    for box, score, scale, angle in raw_candidates:
        if len(selected) >= MAX_TEMPLATE_MATCHES_PER_LOGO:
            break
        if any(_iou(box, other) > NMS_IOU_THRESHOLD for other in occupied):
            continue
        selected.append(
            LogoMatch(
                logo=logo,
                box=box,
                score=score,
                method="template",
                confidence=None,
                scale=scale,
                angle=angle,
            )
        )
        occupied.append(box)

    return selected


def detect_logos_on_frame(
    frame_bgr: np.ndarray,
    logos: Sequence[LogoTemplate],
    model,
    confidence_threshold: float,
    histogram_threshold: float,
    template_threshold: float,
) -> list[LogoMatch]:
    """Находит логотипы на кадре, комбинируя детекции YOLOE и шаблонный поиск."""

    detections = detect_with_yolo(frame_bgr, model, confidence_threshold)
    matches: list[LogoMatch] = []
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    for box, confidence in detections:
        x1, y1, x2, y2 = box
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        best_logo: LogoTemplate | None = None
        best_similarity = -1.0
        for logo in logos:
            similarity = _histogram_similarity(patch, logo)
            if similarity > best_similarity:
                best_similarity = similarity
                best_logo = logo
        if best_logo is None or best_similarity < histogram_threshold:
            continue
        matches.append(
            LogoMatch(
                logo=best_logo,
                box=box,
                score=best_similarity,
                method="yoloe+hist",
                confidence=confidence,
                scale=None,
                angle=None,
            )
        )

    existing_boxes = [match.box for match in matches]
    for logo in logos:
        template_matches = _template_match_multi(frame_gray, logo, template_threshold, existing_boxes)
        for match in template_matches:
            matches.append(match)
            existing_boxes.append(match.box)

    return matches


def draw_logo_matches(frame_bgr: np.ndarray, matches: Sequence[LogoMatch]) -> np.ndarray:
    """Накладывает найденные логотипы на кадр."""

    if not matches:
        return frame_bgr

    overlay = frame_bgr.copy()
    for match in matches:
        x1, y1, x2, y2 = match.box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), match.logo.color, thickness=-1)

    blended = cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0)

    for match in matches:
        x1, y1, x2, y2 = match.box
        color = match.logo.color
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=2)
        label = match.logo.display_name
        extra_parts: list[str] = []
        if match.method == "yoloe+hist":
            extra_parts.append(f"sim={match.score:.2f}")
            if match.confidence is not None:
                extra_parts.append(f"conf={match.confidence:.2f}")
        else:
            extra_parts.append(f"score={match.score:.2f}")
            if match.scale is not None:
                extra_parts.append(f"scale={match.scale:.2f}x")
            if match.angle is not None and abs(match.angle) > 1e-3:
                extra_parts.append(f"angle={match.angle:+.0f}°")
        if extra_parts:
            label = f"{label} ({', '.join(extra_parts)})"

        text_origin = (x1, max(20, y1 - 10))
        cv2.putText(
            blended,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return blended


def mux_audio(
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    reencode: bool = True,
) -> None:
    """Объединяет обработанное видео с оригинальной аудиодорожкой."""

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


def process_video(job: VideoJob, model) -> None:
    """Обрабатывает видео покадрово и сохраняет результат."""

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
    max_frames = total_frames
    if job.seconds is not None and fps > 0:
        max_frames = min(total_frames, int(fps * job.seconds))

    job.intermediate_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(job.intermediate_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать видеофайл: {job.intermediate_video}")

    progress = tqdm(total=max_frames or None, desc=video_path.name, unit="кадр")
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index >= max_frames:
                break

            matches = detect_logos_on_frame(
                frame,
                job.logos,
                model,
                job.confidence_threshold,
                job.histogram_threshold,
                job.template_threshold,
            )
            annotated = draw_logo_matches(frame, matches)
            writer.write(annotated)
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    if job.merge_audio:
        job.output_video.parent.mkdir(parents=True, exist_ok=True)
        mux_audio(job.intermediate_video, video_path, job.output_video)
        if not job.keep_intermediate and job.intermediate_video.exists():
            job.intermediate_video.unlink()
    else:
        job.output_video.parent.mkdir(parents=True, exist_ok=True)
        job.intermediate_video.rename(job.output_video)


def build_jobs(args: argparse.Namespace) -> list[VideoJob]:
    """Готовит задачи обработки на основе аргументов CLI."""

    input_dir: Path = args.input_dir
    logos_root: Path = args.logos_dir
    output_root: Path = args.output_dir

    video_paths: Iterable[Path]
    if args.video is not None:
        video_paths = [args.video]
    else:
        pattern = args.pattern
        video_paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())

    jobs: list[VideoJob] = []
    for video_path in video_paths:
        logo_dir = logos_root / video_path.stem
        try:
            if logo_dir.is_dir() or logo_dir.is_file():
                logos = load_logo_templates(logo_dir)
            else:
                candidate_files = sorted(
                    p
                    for p in logos_root.glob(f"{video_path.stem}*")
                    if p.is_file()
                )
                if not candidate_files:
                    raise FileNotFoundError(
                        f"каталог или файл с логотипами {logo_dir} не найден"
                    )
                logos = _prepare_logo_templates(candidate_files)
        except FileNotFoundError as error:
            print(f"Пропускаю {video_path.name}: {error}")
            continue
        except RuntimeError as error:
            print(f"Пропускаю {video_path.name}: {error}")
            continue

        output_video = output_root / f"{video_path.stem}{DEFAULT_OUTPUT_SUFFIX}"
        intermediate_video = output_root / f"{video_path.stem}{DEFAULT_INTERMEDIATE_SUFFIX}"
        jobs.append(
            VideoJob(
                video_path=video_path,
                logos=logos,
                output_video=output_video,
                intermediate_video=intermediate_video,
                seconds=args.seconds,
                confidence_threshold=args.confidence,
                histogram_threshold=args.hist_threshold,
                template_threshold=args.template_threshold,
                keep_intermediate=args.keep_intermediate,
                merge_audio=not args.no_audio,
            )
        )

    if not jobs:
        raise RuntimeError("Не найдено ни одного видео для обработки")
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Конкретный видеофайл для обработки")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Каталог с исходными видео",
    )
    parser.add_argument(
        "--logos-dir",
        type=Path,
        default=DEFAULT_LOGO_DIR,
        help="Каталог с логотипами, организованный по именам видео",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для сохранения обработанных видео",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Глоб-шаблон для поиска видео при пакетной обработке",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Имя или путь к весам модели YOLOE",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Ограничить обработку первыми N секундами видео",
    )
    parser.add_argument(
        "--conf",
        dest="confidence",
        type=float,
        default=0.25,
        help="Порог уверенности YOLOE",
    )
    parser.add_argument(
        "--hist-threshold",
        type=float,
        default=0.6,
        help="Минимальная корреляция гистограмм для подтверждения логотипа",
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.55,
        help="Порог совпадения при шаблонном поиске",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Не удалять промежуточное видео без аудио",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Не объединять результат с исходной аудиодорожкой",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    model, _ = load_yoloe_model(args.model)

    for job in jobs:
        print(f"Обрабатываю {job.video_path.name}...")
        process_video(job, model)

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
