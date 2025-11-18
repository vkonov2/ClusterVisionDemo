"""Генерация видео с подсветкой логотипа по визуальному промпту YOLOE."""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from tqdm.auto import tqdm

from detect_logo_last_frame import extract_last_frame
from generate_yoloe_logo_overlay_video import (
    DEFAULT_MODEL,
    LogoMatch,
    LogoTemplate,
    detect_logos_on_frame,
    load_logo_templates,
    load_yoloe_model,
    mux_audio,
)


DEFAULT_VIDEO = Path("data/videos/010-30.mp4")
DEFAULT_LOGO = Path("data/logos/010-30.png")
DEFAULT_OUTPUT = Path("outputs/logos/010-30-prompt.mp4")
DEFAULT_INTERMEDIATE = Path("outputs/logos/010-30-prompt-silent.mp4")

DEFAULT_CONFIDENCE = 0.1
DEFAULT_PROMPT_THRESHOLD = 0.55
DEFAULT_HIST_THRESHOLD = 0.4
DEFAULT_TEMPLATE_THRESHOLD = 0.45

PROMPT_PREVIEW_SIZE = (128, 128)


@dataclass(frozen=True)
class PromptRegion:
    """Описание области-промпта, извлечённой с последнего кадра."""

    name: str
    box: tuple[int, int, int, int]
    image_bgr: np.ndarray
    resized_bgr: np.ndarray
    resized_gray: np.ndarray
    histogram: np.ndarray

    @property
    def color(self) -> tuple[int, int, int]:
        return _color_from_name(self.name)


@dataclass(frozen=True)
class PromptMatch:
    """Результат сопоставления промпта с кадром."""

    prompt: PromptRegion
    box: tuple[int, int, int, int]
    score: float
    confidence: float
    hist_score: float
    template_score: float
    l1_score: float


# локальная копия функции из generate_yoloe_logo_overlay_video
# (импортировать приватную функцию напрямую не хотелось бы)
def _color_from_name(name: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(name.encode("utf-8"), usedforsecurity=False).digest()
    r = digest[0]
    g = digest[1]
    b = digest[2]
    return int(b // 2 + 96), int(g // 2 + 96), int(r // 2 + 96)


def _compute_histogram(image_bgr: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _prepare_prompt(match: LogoMatch, frame: np.ndarray) -> PromptRegion:
    x1, y1, x2, y2 = match.box
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Пустой кроп для промпта")

    resized = cv2.resize(crop, PROMPT_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    histogram = _compute_histogram(crop)

    return PromptRegion(
        name=match.logo.display_name,
        box=(x1, y1, x2, y2),
        image_bgr=crop,
        resized_bgr=resized,
        resized_gray=resized_gray,
        histogram=histogram,
    )


def _prepare_prompts(
    frame: np.ndarray,
    logos: Sequence[LogoTemplate],
    model,
    confidence_threshold: float,
    hist_threshold: float,
    template_threshold: float,
) -> list[PromptRegion]:
    matches = detect_logos_on_frame(
        frame,
        logos,
        model,
        confidence_threshold,
        hist_threshold,
        template_threshold,
    )
    if not matches:
        raise RuntimeError(
            "Не удалось выделить промпт: на последнем кадре не найдено ни одного логотипа"
        )

    prompts: list[PromptRegion] = []
    for match in matches:
        try:
            prompts.append(_prepare_prompt(match, frame))
        except ValueError:
            continue

    if not prompts:
        raise RuntimeError("Получены некорректные координаты промпта")

    return prompts


def _clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width)))
    y2 = int(max(0, min(y2, height)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _compute_match_scores(prompt: PromptRegion, candidate: np.ndarray) -> tuple[float, float, float, float]:
    if candidate.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    resized = cv2.resize(candidate, PROMPT_PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    candidate_hist = _compute_histogram(candidate)
    hist_score = cv2.compareHist(prompt.histogram, candidate_hist, cv2.HISTCMP_CORREL)
    hist_score = (hist_score + 1.0) / 2.0

    candidate_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    template_value = cv2.matchTemplate(candidate_gray, prompt.resized_gray, cv2.TM_CCOEFF_NORMED)
    template_score = float(template_value.max())
    template_score = (template_score + 1.0) / 2.0

    diff = cv2.absdiff(resized, prompt.resized_bgr)
    l1_score = 1.0 - float(diff.mean()) / 255.0

    total_score = 0.5 * hist_score + 0.3 * template_score + 0.2 * l1_score
    return float(total_score), float(hist_score), float(template_score), float(l1_score)


def _match_prompts_on_frame(
    frame: np.ndarray,
    prompts: Sequence[PromptRegion],
    yolo_results,
    score_threshold: float,
) -> list[PromptMatch]:
    matches: list[PromptMatch] = []
    if not prompts:
        return matches

    if not yolo_results:
        return matches

    result = yolo_results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return matches

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    height, width = frame.shape[:2]

    for coords, confidence in zip(xyxy, confs):
        x1, y1, x2, y2 = coords.astype(int)
        x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, width, height)
        candidate = frame[y1:y2, x1:x2]
        if candidate.size == 0:
            continue

        best_prompt: PromptRegion | None = None
        best_score = -math.inf
        best_hist = 0.0
        best_template = 0.0
        best_l1 = 0.0

        for prompt in prompts:
            score, hist_score, template_score, l1_score = _compute_match_scores(prompt, candidate)
            if score > best_score:
                best_score = score
                best_prompt = prompt
                best_hist = hist_score
                best_template = template_score
                best_l1 = l1_score

        if best_prompt is None:
            continue

        if best_score < score_threshold:
            continue

        matches.append(
            PromptMatch(
                prompt=best_prompt,
                box=(x1, y1, x2, y2),
                score=float(best_score),
                confidence=float(confidence),
                hist_score=float(best_hist),
                template_score=float(best_template),
                l1_score=float(best_l1),
            )
        )

    return matches


def _draw_matches(frame: np.ndarray, matches: Sequence[PromptMatch]) -> np.ndarray:
    annotated = frame.copy()
    for match in matches:
        x1, y1, x2, y2 = match.box
        color = match.prompt.color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{match.prompt.name}: score={match.score:.2f} | conf={match.confidence:.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def process_video(
    video_path: Path,
    prompts: Sequence[PromptRegion],
    model,
    output_path: Path,
    intermediate_path: Path,
    seconds: float | None,
    confidence_threshold: float,
    prompt_threshold: float,
    keep_intermediate: bool,
    merge_audio_track: bool,
) -> None:
    video_path = video_path.expanduser()
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
    if seconds is not None and fps > 0:
        max_frames = min(total_frames, int(fps * seconds))

    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать видеофайл: {intermediate_path}")

    progress = tqdm(total=max_frames or None, desc=video_path.name, unit="кадр")

    frame_index = 0
    try:
        while True:
            success, frame = cap.read()
            if not success or frame_index >= max_frames:
                break

            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            matches = _match_prompts_on_frame(frame, prompts, results, prompt_threshold)
            annotated = _draw_matches(frame, matches)

            writer.write(annotated)
            frame_index += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()

    if merge_audio_track:
        mux_audio(intermediate_path, video_path, output_path)
        if not keep_intermediate and intermediate_path.exists():
            intermediate_path.unlink()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_path.rename(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Видео для обработки")
    parser.add_argument("--logo", type=Path, default=DEFAULT_LOGO, help="Логотип для детекции")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Выходной файл с подсвеченным логотипом",
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=DEFAULT_INTERMEDIATE,
        help="Путь до промежуточного файла без аудио",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Веса модели YOLOE")
    parser.add_argument(
        "--seconds",
        type=float,
        help="Сколько первых секунд видео обрабатывать (по умолчанию всё видео)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Минимальная уверенность YOLOE для кандидатов",
    )
    parser.add_argument(
        "--prompt-threshold",
        type=float,
        default=DEFAULT_PROMPT_THRESHOLD,
        help="Минимальный итоговый скор соответствия промпту",
    )
    parser.add_argument(
        "--hist-threshold",
        type=float,
        default=DEFAULT_HIST_THRESHOLD,
        help="Порог гистограмм для поиска промпта на последнем кадре",
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=DEFAULT_TEMPLATE_THRESHOLD,
        help="Порог шаблонного поиска для промпта на последнем кадре",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Не удалять промежуточный файл без аудио",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Не объединять видео с оригинальной аудиодорожкой",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path: Path = args.video
    logo_path: Path = args.logo
    output_path: Path = args.output
    intermediate_path: Path = args.intermediate

    logos = load_logo_templates(logo_path)
    model, _ = load_yoloe_model(args.model)

    last_frame = extract_last_frame(video_path)
    prompts = _prepare_prompts(
        last_frame,
        logos,
        model,
        args.confidence,
        args.hist_threshold,
        args.template_threshold,
    )

    process_video(
        video_path,
        prompts,
        model,
        output_path,
        intermediate_path,
        args.seconds,
        args.confidence,
        args.prompt_threshold,
        args.keep_intermediate,
        not args.no_audio,
    )

    print(f"Результат сохранён в {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
