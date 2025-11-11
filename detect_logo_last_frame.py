"""Поиск логотипа на последнем кадре видео с подсветкой результата."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from generate_yoloe_logo_overlay_video import (
    DEFAULT_MODEL,
    LogoMatch,
    LogoTemplate,
    detect_logos_on_frame,
    draw_logo_matches,
    load_logo_templates,
    load_yoloe_model,
)


DEFAULT_VIDEO = Path("data/videos/010-30.mp4")
DEFAULT_LOGO = Path("data/logos/010-30.png")
DEFAULT_OUTPUT = Path("outputs/logos/010-30-last-frame.png")


def extract_last_frame(video_path: Path) -> np.ndarray:
    """Возвращает последний кадр видео."""

    video_path = video_path.expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = False
    frame: np.ndarray | None = None

    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames - 1, 0))
        success, frame = cap.read()

    if not success:
        # если не удалось прочитать напрямую, читаем последовательным перебором
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            next_success, next_frame = cap.read()
            if not next_success:
                break
            success = True
            frame = next_frame

    cap.release()

    if not success or frame is None:
        raise RuntimeError("Не удалось получить последний кадр видео")

    return frame


def annotate_last_frame(
    frame: np.ndarray,
    logos: Sequence[LogoTemplate],
    model,
    confidence_threshold: float,
    histogram_threshold: float,
    template_threshold: float,
) -> tuple[np.ndarray, list[LogoMatch]]:
    """Ищет логотипы на кадре и возвращает изображение с подсветкой."""

    matches = detect_logos_on_frame(
        frame,
        logos,
        model,
        confidence_threshold,
        histogram_threshold,
        template_threshold,
    )
    annotated = draw_logo_matches(frame, matches)
    return annotated, list(matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Путь к видеофайлу (по умолчанию data/videos/010-30.mp4)",
    )
    parser.add_argument(
        "--logo",
        type=Path,
        default=DEFAULT_LOGO,
        help="Путь к изображению логотипа (по умолчанию data/logos/010-30.png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Куда сохранить кадр с подсветкой найденных логотипов",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Имя или путь к весам модели YOLOE",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    frame = extract_last_frame(args.video)
    logos = load_logo_templates(args.logo)
    model, _ = load_yoloe_model(args.model)

    annotated, matches = annotate_last_frame(
        frame,
        logos,
        model,
        args.confidence,
        args.hist_threshold,
        args.template_threshold,
    )

    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    if matches:
        print("Найдены совпадения:")
        for match in matches:
            x1, y1, x2, y2 = match.box
            parts = [
                f"{match.logo.display_name}",
                f"метод={match.method}",
                f"score={match.score:.3f}",
            ]
            if match.confidence is not None:
                parts.append(f"conf={match.confidence:.3f}")
            if match.scale is not None:
                parts.append(f"scale={match.scale:.2f}x")
            parts.append(f"bbox=({x1}, {y1}, {x2}, {y2})")
            print(" - " + ", ".join(parts))
    else:
        print("Совпадения не найдены")

    print(f"Кадр сохранён в {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
