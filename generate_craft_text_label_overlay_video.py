"""Выделение текстовых областей, соответствующих названию ролика."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import requests
import torch
from tqdm.auto import tqdm


CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "craft"
REPO_DIR = CACHE_DIR / "repo"
REPO_URL = "https://github.com/clovaai/CRAFT-pytorch.git"
WEIGHTS_PATH = CACHE_DIR / "weights" / "craft_mlt_25k.pth"
WEIGHTS_FILE_ID = "1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"

DEFAULT_VIDEOS_DIR = Path("data/videos")
DEFAULT_OUTPUT_DIR = Path("outputs/craft_text_labels")
DEFAULT_TEXTS_FILE = Path("data/texts/texts.json")


@dataclass(frozen=True)
class LabeledBox:
    """Информация о найденной текстовой области."""

    contour: np.ndarray
    label: str
    recognized: str


@dataclass(frozen=True)
class LabelTemplate:
    """Шаблон текстовой подписи для сопоставления с детектированными боксами."""

    label: str
    token: str
    image: np.ndarray
    inverted: np.ndarray
    threshold: float


def _render_text_template(
    text: str,
    scale: float,
    thickness: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray | None:
    """Создаёт шаблон текста в градациях серого."""

    text = text.strip()
    if not text:
        return None

    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    width = max(width, 1)
    height = max(height, 1)
    canvas = np.zeros((height + baseline + 4, width + 4), dtype=np.uint8)
    origin = (2, height + 2)
    cv2.putText(canvas, text, origin, font, scale, 255, thickness, lineType=cv2.LINE_AA)

    normalized = canvas.astype(np.float32) / 255.0
    if float(normalized.max()) <= 0.0:
        return None
    return normalized


def build_label_templates(labels: Sequence[str]) -> list[LabelTemplate]:
    """Готовит набор шаблонов для поисковых фраз."""

    templates: list[LabelTemplate] = []
    seen_keys: set[tuple[str, str, int, int]] = set()

    for label in labels:
        raw = label.strip()
        if not raw:
            continue

        tokens = {raw}
        tokens.update(token for token in re.split(r"[^0-9A-Za-zА-Яа-яЁё]+", raw) if token)

        for token in tokens:
            normalized_token = token.strip()
            if not normalized_token:
                continue

            for scale in (0.8, 1.0, 1.2, 1.5):
                for thickness in (1, 2):
                    rendered = _render_text_template(normalized_token, scale=scale, thickness=thickness)
                    if rendered is None:
                        continue

                    key = (label, normalized_token, rendered.shape[0], rendered.shape[1])
                    if key in seen_keys:
                        continue

                    seen_keys.add(key)
                    inverted = 1.0 - rendered
                    threshold = 0.55 if len(normalized_token) >= 4 else 0.65
                    templates.append(
                        LabelTemplate(
                            label=label,
                            token=normalized_token,
                            image=rendered,
                            inverted=inverted,
                            threshold=threshold,
                        )
                    )

    return templates


def ensure_craft_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Клонирует репозиторий CRAFT и применяет патчи совместимости."""

    repo_dir = repo_dir.expanduser().resolve()
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Клонирую CRAFT в {repo_dir}...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

    patch_craft_repo_for_torchvision(repo_dir)
    return repo_dir


def patch_craft_repo_for_torchvision(repo_dir: Path) -> None:
    """Добавляет заглушку model_urls для современных версий torchvision."""

    target = repo_dir / "basenet" / "vgg16_bn.py"
    if not target.exists():
        return

    marker = "# torchvision>=0.15 compatibility"
    contents = target.read_text(encoding="utf-8")
    if marker in contents:
        return

    needle = "from torchvision.models.vgg import model_urls"
    if needle not in contents:
        return

    replacement = (
        "try:\n"
        "    from torchvision.models.vgg import model_urls  # type: ignore\n"
        "except ImportError:  # torchvision>=0.15 compatibility\n"
        "    from torchvision.models import VGG16_BN_Weights\n"
        "    model_urls = {\"vgg16_bn\": VGG16_BN_Weights.IMAGENET1K_V1.url}\n"
        f"{marker}\n"
    )

    target.write_text(contents.replace(needle, replacement, 1), encoding="utf-8")


def ensure_craft_weights(weights_path: Path = WEIGHTS_PATH) -> Path:
    """Скачивает предобученные веса детектора текста."""

    weights_path = weights_path.expanduser().resolve()
    if weights_path.exists():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    print("Скачиваю веса CRAFT из Google Drive...")
    download_file_from_google_drive(WEIGHTS_FILE_ID, weights_path)
    if not weights_path.exists():
        raise RuntimeError("Не удалось скачать веса модели CRAFT")
    return weights_path


def download_file_from_google_drive(file_id: str, destination: Path, chunk_size: int = 32768) -> None:
    """Загружает файл из Google Drive без сторонних зависимостей."""

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
    """Ищет токен подтверждения скачивания крупного файла."""

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response: requests.Response, destination: Path, chunk_size: int) -> None:
    """Сохраняет потоковый ответ в файл с временным буфером."""

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


def copy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Удаляет префикс 'module.' из ключей state_dict, если он присутствует."""

    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if parts[0] == "module":
            parts = parts[1:]
        new_state_dict[".".join(parts)] = value
    return new_state_dict


def load_craft_model(device: torch.device | None = None) -> tuple[torch.nn.Module, any]:
    """Загружает модель CRAFT и необходимые постпроцессоры."""

    repo_dir = ensure_craft_repo()
    weights_path = ensure_craft_weights()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from craft import CRAFT  # type: ignore  # noqa: WPS433
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

    resized = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    target_h32 = target_h if target_h % 32 == 0 else target_h + (32 - target_h % 32)
    target_w32 = target_w if target_w % 32 == 0 else target_w + (32 - target_w % 32)

    canvas = np.zeros((target_h32, target_w32, 3), dtype=np.float32)
    canvas[0:target_h, 0:target_w, :] = resized
    return canvas, ratio


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


def draw_labeled_boxes(
    frame_bgr: np.ndarray,
    labeled_boxes: Iterable[LabeledBox],
    fill_alpha: float = 0.35,
) -> np.ndarray:
    """Рисует подсветку и подписи для обнаруженных областей."""

    labeled_boxes = list(labeled_boxes)
    if not labeled_boxes:
        return frame_bgr

    overlay = frame_bgr.copy()
    for item in labeled_boxes:
        contour = item.contour.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [contour], (0, 165, 255))

    blended = cv2.addWeighted(overlay, fill_alpha, frame_bgr, 1.0 - fill_alpha, 0)

    for item in labeled_boxes:
        contour = item.contour.reshape(-1, 2).astype(np.int32)
        cv2.polylines(blended, [contour], True, (0, 255, 255), thickness=2)

        x_min = int(contour[:, 0].min())
        y_min = int(contour[:, 1].min())
        label = f"{item.label}: {item.recognized.strip()}"
        cv2.putText(
            blended,
            label,
            (x_min, max(20, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            label,
            (x_min, max(20, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return blended


def _extract_box_roi(frame_bgr: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
    """Возвращает вырезанный по контуру регион интереса."""

    contour = contour.reshape(-1, 2)
    x_coords = contour[:, 0]
    y_coords = contour[:, 1]

    x_min = max(int(math.floor(x_coords.min())) - 2, 0)
    y_min = max(int(math.floor(y_coords.min())) - 2, 0)
    x_max = min(int(math.ceil(x_coords.max())) + 2, frame_bgr.shape[1])
    y_max = min(int(math.ceil(y_coords.max())) + 2, frame_bgr.shape[0])

    if x_max <= x_min or y_max <= y_min:
        return None

    roi = frame_bgr[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    return roi


def _prepare_roi_image(roi_bgr: np.ndarray) -> np.ndarray:
    """Готовит изображение ROI для шаблонного сравнения."""

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = cv2.normalize(blurred.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized


def _match_template(search: np.ndarray, template: np.ndarray) -> float:
    """Сравнивает ROI с шаблоном и возвращает степень совпадения."""

    if search.shape[0] < 2 or search.shape[1] < 2:
        return 0.0

    tmpl = template
    search_img = search

    if search_img.shape[0] < tmpl.shape[0] or search_img.shape[1] < tmpl.shape[1]:
        tmpl = cv2.resize(tmpl, (search_img.shape[1], search_img.shape[0]), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
        return float(result[0, 0])

    result = cv2.matchTemplate(search_img, tmpl, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def _find_best_template(
    roi: np.ndarray,
    templates: Sequence[LabelTemplate],
) -> tuple[LabelTemplate, float] | None:
    """Подбирает шаблон с максимальным совпадением."""

    best_score = 0.0
    best_template: LabelTemplate | None = None

    for template in templates:
        score_direct = _match_template(roi, template.image)
        score_inverted = _match_template(roi, template.inverted)
        score = max(score_direct, score_inverted)
        if score > best_score:
            best_score = score
            best_template = template

    if best_template is None:
        return None

    return best_template, best_score


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
def load_video_text_labels(texts_path: Path) -> dict[str, list[str]]:
    """Загружает сопоставление роликов и поисковых фраз."""

    texts_path = texts_path.expanduser().resolve()
    if not texts_path.exists():
        raise FileNotFoundError(
            f"Файл с текстовыми подсказками не найден: {texts_path}"
        )

    payload = json.loads(texts_path.read_text(encoding="utf-8"))
    mapping: dict[str, list[str]] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            phrases: list[str]
            if isinstance(value, str):
                phrases = [value]
            elif isinstance(value, Sequence):
                phrases = [str(item) for item in value]
            else:
                continue
            mapping[str(key)] = [phrase for phrase in phrases if phrase.strip()]

    return mapping


def select_matching_boxes(
    frame_bgr: np.ndarray,
    boxes: Iterable[np.ndarray],
    templates: Sequence[LabelTemplate],
) -> list[LabeledBox]:
    """Фильтрует боксы по совпадению с шаблонами текстовых подсказок."""

    templates = [template for template in templates if template is not None]
    if not templates:
        return []

    matches: list[LabeledBox] = []
    for contour in boxes:
        roi_bgr = _extract_box_roi(frame_bgr, contour)
        if roi_bgr is None:
            continue

        roi = _prepare_roi_image(roi_bgr)
        result = _find_best_template(roi, templates)
        if result is None:
            continue

        template, score = result
        if score < template.threshold:
            continue

        recognized = f"{template.token} ({score:.2f})"
        matches.append(LabeledBox(contour=contour, label=template.label, recognized=recognized))

    return matches


def iter_video_frames(video_path: Path) -> tuple[cv2.VideoCapture, int, int, float]:
    """Возвращает видеочитатель, размеры и FPS."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    return capture, width, height, fps


def process_video(
    video_path: Path,
    output_video: Path,
    model: torch.nn.Module,
    craft_utils_module,
    device: torch.device,
    templates: Sequence[LabelTemplate],
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    canvas_size: int,
    mag_ratio: float,
) -> None:
    """Обрабатывает отдельный ролик и сохраняет итоговое видео."""

    output_video.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output_video.with_suffix(".silent.mp4")

    capture, width, height, fps = iter_video_frames(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(intermediate), fourcc, fps, (width, height))

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    with tqdm(total=total_frames, desc=video_path.name, unit="кадр") as progress:
        while True:
            success, frame = capture.read()
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

            labeled = select_matching_boxes(frame, boxes, templates)
            overlay = draw_labeled_boxes(frame, labeled)
            writer.write(overlay)
            progress.update(1)

    capture.release()
    writer.release()

    mux_audio(intermediate, video_path, output_video)
    intermediate.unlink(missing_ok=True)


def gather_videos(videos_dir: Path, single_video: Path | None) -> list[Path]:
    """Подготавливает список роликов для обработки."""

    if single_video is not None:
        return [single_video]

    videos: list[Path] = []
    for pattern in ("*.mp4", "*.mov", "*.mkv"):
        videos.extend(sorted(videos_dir.glob(pattern)))
    return videos


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Поиск текста на каждом кадре с помощью CRAFT и подсветка областей,"
            " содержащих целевые фразы из texts.json"
        )
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Каталог с исходными видео",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Обработать только указанный файл",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для сохранения результатов",
    )
    parser.add_argument(
        "--texts",
        type=Path,
        default=DEFAULT_TEXTS_FILE,
        help="JSON-файл с текстовыми подсказками",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.7,
        help="Порог вероятности текста для CRAFT",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.4,
        help="Порог связи символов для CRAFT",
    )
    parser.add_argument(
        "--low-text",
        type=float,
        default=0.4,
        help="Минимальный порог текстовой карты для фильтрации",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=2048,
        help="Размер холста для масштабирования входа",
    )
    parser.add_argument(
        "--mag-ratio",
        type=float,
        default=1.5,
        help="Коэффициент увеличения изображения перед подачей в модель",
    )
    args = parser.parse_args(argv)

    videos_dir = args.videos_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    texts_path = args.texts.expanduser().resolve()

    video_labels = load_video_text_labels(texts_path)

    target_videos = gather_videos(videos_dir, args.video.expanduser().resolve() if args.video else None)
    if not target_videos:
        raise RuntimeError("Не найдено ни одного видео для обработки")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, craft_utils_module = load_craft_model(device)
    for video_path in target_videos:
        key = video_path.stem
        labels = video_labels.get(key)
        if not labels:
            print(f"Пропускаю {video_path.name}: нет текстовой подписи в {texts_path}")
            continue

        templates = build_label_templates(labels)
        if not templates:
            print(f"Пропускаю {video_path.name}: не удалось подготовить шаблоны текста")
            continue

        output_video = output_dir / f"{video_path.stem}-text-labels.mp4"
        process_video(
            video_path=video_path,
            output_video=output_video,
            model=model,
            craft_utils_module=craft_utils_module,
            device=device,
            templates=templates,
            text_threshold=args.text_threshold,
            link_threshold=args.link_threshold,
            low_text=args.low_text,
            canvas_size=args.canvas_size,
            mag_ratio=args.mag_ratio,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
