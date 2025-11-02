"""Выделение текста на видео с помощью модели CRAFT."""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "craft"
REPO_DIR = CACHE_DIR / "repo"
REPO_URL = "https://github.com/clovaai/CRAFT-pytorch.git"
WEIGHTS_PATH = CACHE_DIR / "weights" / "craft_mlt_25k.pth"
WEIGHTS_FILE_ID = "1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"

DEFAULT_VIDEO = Path("data/videos/000-youtube.mp4")
DEFAULT_OUTPUT = Path("outputs/craft/000-youtube-text.mp4")
DEFAULT_INTERMEDIATE = Path("outputs/craft/000-youtube-text-silent.mp4")


def ensure_craft_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Клонирует репозиторий CRAFT при первом запуске."""

    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Клонирую CRAFT в {repo_dir}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def ensure_craft_weights(weights_path: Path = WEIGHTS_PATH) -> Path:
    """Скачивает предобученные веса детектора текста."""

    weights_path = weights_path.expanduser().resolve()
    if weights_path.exists():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    print("Скачиваю веса CRAFT из Google Drive...")
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - информативное сообщение
        raise RuntimeError(
            "Пакет gdown не установлен. Добавьте его в окружение: pip install gdown"
        ) from exc

    url = f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}"
    gdown.download(url, str(weights_path), quiet=False)
    if not weights_path.exists():
        raise RuntimeError("Не удалось скачать веса модели CRAFT")
    return weights_path


def copy_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Удаляет префикс 'module.' из ключей state_dict, если он присутствует."""

    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        parts = key.split(".")
        if parts[0] == "module":
            parts = parts[1:]
        new_key = ".".join(parts)
        new_state_dict[new_key] = value
    return new_state_dict


def load_craft_model(device: torch.device | None = None) -> tuple[torch.nn.Module, Any]:
    """Загружает модель CRAFT и необходимые постпроцессоры."""

    repo_dir = ensure_craft_repo()
    weights_path = ensure_craft_weights()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from craft import CRAFT  # type: ignore  # noqa: WPS433 - динамическая подгрузка из репозитория
    import craft_utils  # type: ignore  # noqa: WPS433

    model = CRAFT()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(copy_state_dict(state_dict))
    model.to(device)
    model.eval()

    return model, craft_utils


def normalize_mean_variance(
    img: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    variance: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Нормализация входного RGB-изображения как в оригинальном коде CRAFT."""

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
    """Изменяет размер кадра, сохраняя пропорции, аналогично оригинальной реализации."""

    height, width, _ = img.shape
    target_size = mag_ratio * max(height, width)
    if target_size > square_size:
        target_size = float(square_size)

    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    target_h32 = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
    target_w32 = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, 3), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    return resized, ratio


def detect_text_boxes(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    craft_utils_module,
    device: torch.device,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    canvas_size: int,
    mag_ratio: float,
) -> list[np.ndarray]:
    """Возвращает список найденных прямоугольников в кадре."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized, target_ratio = resize_aspect_ratio(frame_rgb, canvas_size, mag_ratio=mag_ratio)
    ratio = 1 / target_ratio if target_ratio > 0 else 1.0

    norm = normalize_mean_variance(resized)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _ = model(tensor)

    score_text = y[0, :, :, 0].detach().cpu().numpy()
    score_link = y[0, :, :, 1].detach().cpu().numpy()

    boxes, _ = craft_utils_module.getDetBoxes(
        score_text,
        score_link,
        text_threshold,
        link_threshold,
        low_text,
        poly=False,
    )

    if not boxes:
        return []

    boxes = craft_utils_module.adjustResultCoordinates(boxes, ratio, ratio)
    return [np.array(box, dtype=np.float32) for box in boxes]


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: Iterable[np.ndarray],
    fill_color: tuple[int, int, int] = (0, 255, 0),
    border_color: tuple[int, int, int] = (0, 255, 255),
    fill_alpha: float = 0.25,
    border_thickness: int = 2,
) -> np.ndarray:
    """Наносит прямоугольники на кадр с лёгким подсвечиванием."""

    if not boxes:
        return frame_bgr

    overlay = frame_bgr.copy()
    contours: list[np.ndarray] = []
    for box in boxes:
        contour = box.reshape(-1, 2).astype(np.int32)
        contours.append(contour)
        cv2.fillPoly(overlay, [contour], fill_color)

    blended = cv2.addWeighted(overlay, fill_alpha, frame_bgr, 1.0 - fill_alpha, 0)
    for contour in contours:
        cv2.polylines(blended, [contour], isClosed=True, color=border_color, thickness=border_thickness)
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


def process_video(
    video_path: Path,
    output_video: Path,
    intermediate_video: Path,
    model: torch.nn.Module,
    craft_utils_module,
    device: torch.device,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    canvas_size: int,
    mag_ratio: float,
    keep_intermediate: bool,
) -> None:
    """Обрабатывает видео покадрово и сохраняет результат."""

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    intermediate_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать видеофайл: {intermediate_video}")

    progress = tqdm(total=total_frames or None, desc=video_path.name, unit="кадр")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            boxes = detect_text_boxes(
                frame,
                model=model,
                craft_utils_module=craft_utils_module,
                device=device,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=low_text,
                canvas_size=canvas_size,
                mag_ratio=mag_ratio,
            )
            highlighted = draw_boxes(frame, boxes)
            writer.write(highlighted)
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    mux_audio(intermediate_video, video_path, output_video)

    if not keep_intermediate and intermediate_video.exists():
        intermediate_video.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Выделение текста на видео с помощью CRAFT"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Путь к исходному видео",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Путь к финальному видео с аудио",
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=DEFAULT_INTERMEDIATE,
        help="Временный путь для немого видео с подсветкой",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.7,
        help="Порог для уверенности детекции текста",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        help="Порог для связей между символами",
    )
    parser.add_argument(
        "--low-text",
        type=float,
        default=0.4,
        help="Минимальный порог для карт тепла",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=1280,
        help="Размер холста для изменения масштаба кадра",
    )
    parser.add_argument(
        "--mag-ratio",
        type=float,
        default=1.5,
        help="Коэффициент масштабирования перед подачей в модель",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Явно указать устройство (например, cuda или cpu)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Не удалять немой вариант видео после объединения",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, craft_utils_module = load_craft_model(device)

    process_video(
        video_path=args.video,
        output_video=args.output,
        intermediate_video=args.intermediate,
        model=model,
        craft_utils_module=craft_utils_module,
        device=device,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        low_text=args.low_text,
        canvas_size=args.canvas_size,
        mag_ratio=args.mag_ratio,
        keep_intermediate=args.keep_intermediate,
    )

    print(f"Готово! Финальное видео сохранено в {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
