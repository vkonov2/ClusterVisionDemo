"""Детекция логотипов брендов на видео с помощью Logodetect."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from tqdm.auto import tqdm

from generate_unisal_saliency import read_video_frames


REPO_URL = "https://github.com/Heldenkombinat/Logodetect.git"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "logodetect"
REPO_DIR = CACHE_DIR / "repo"
DEFAULT_VIDEOS_DIR = Path("data/videos")
DEFAULT_LOGOS_DIR = Path("data/logos")
DEFAULT_OUTPUT_DIR = Path("outputs/logo_detection")
DEFAULT_LOGOS_LIST = DEFAULT_LOGOS_DIR / "logos_list.json"


def ensure_logodetect_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Клонирует репозиторий Logodetect в кэш-директорию при необходимости."""

    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Клонируем Logodetect в {repo_dir}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def install_logodetect_dependencies(repo_dir: Path) -> None:
    """Устанавливает зависимости Logodetect."""

    requirements_file = repo_dir / "requirements.txt"
    if not requirements_file.exists():
        print(f"Предупреждение: requirements.txt не найден в {repo_dir}")
        return

    print("Устанавливаем зависимости Logodetect...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        check=True,
    )
    
    # Устанавливаем совместимую версию moviepy (Logodetect требует старый API)
    print("Устанавливаем совместимую версию moviepy...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "moviepy==1.0.3"],
        check=True,
    )
    
    # Устанавливаем совместимые версии numpy и opencv-python
    # imgaug не поддерживает NumPy 2.0, но opencv-python 4.12 требует numpy>=2
    # Используем более старую версию opencv-python, которая работает с numpy 1.x
    # Сначала устанавливаем opencv-python, затем принудительно устанавливаем numpy<2.0
    print("Устанавливаем совместимые версии opencv-python и numpy...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "opencv-python<4.10"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy<2.0", "--force-reinstall"],
        check=True,
    )


def download_logodetect_models(data_dir: Path) -> None:
    """Скачивает модели Logodetect, если они отсутствуют."""
    
    BASE_URL = "https://hkt-logodetect.s3.eu-central-1.amazonaws.com"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_download = [
        ("detector.pth", "model"),
        ("embedder.pth", "model"),
        ("classifier_resnet18.pth", "model"),
    ]
    
    for file_name, data_type in models_to_download:
        if data_type == "model":
            local_path = models_dir / file_name
        else:
            local_path = data_dir / file_name
        
        if local_path.exists():
            print(f"Модель {file_name} уже существует, пропускаем...")
            continue
        
        print(f"Скачиваем {file_name}...")
        url = f"{BASE_URL}/{file_name}"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            local_path.write_bytes(response.content)
            print(f"✓ {file_name} загружен")
        except Exception as e:
            print(f"Ошибка при загрузке {file_name}: {e}")
            raise


def load_logodetect_recognizer(repo_dir: Path, exemplars_dir: Path, device: str, logo_filename: str) -> object:
    """Загружает Recognizer из Logodetect с поддержкой one-shot detection."""
    
    sys.path.insert(0, str(repo_dir))
    
    # Импортируем необходимые модули
    try:
        from logodetect.recognizer import Recognizer  # type: ignore
        from logodetect.utils import clean_name  # type: ignore
        import logodetect.constants as constants  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Не удалось импортировать модули Logodetect. "
            f"Убедитесь, что зависимости установлены: {e}"
        )

    # Устанавливаем переменную окружения для данных
    data_dir = Path.home() / ".hkt" / "logodetect"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Скачиваем модели, если их нет
    models_dir = data_dir / "models"
    if not models_dir.exists() or not any(models_dir.glob("*.pth")):
        print("Модели Logodetect не найдены, скачиваем...")
        download_logodetect_models(data_dir)

    os.environ.setdefault("LOGOS_RECOGNITION", str(data_dir))

    # Определяем имя бренда из имени файла логотипа
    brand_name = clean_name(logo_filename)
    
    # Добавляем наш бренд в BRAND_LOGOS для фильтрации exemplars
    # Сохраняем оригинальное значение и восстанавливаем после создания Recognizer
    original_brand_logos = constants.BRAND_LOGOS.copy()
    if brand_name not in constants.BRAND_LOGOS:
        constants.BRAND_LOGOS.append(brand_name)
        print(f"Добавлен бренд '{brand_name}' в список для детекции")

    # Создаем конфигурацию с включенным классификатором для one-shot detection
    config = {
        "DETECTOR_DEVICE": device,
        "CLASSIFIER_DEVICE": device,
        "EMBEDDER_DEVICE": device,
        "DEVICE": device,
        "USE_CLASSIFIER": True,  # Включаем классификатор для one-shot detection
        "EXEMPLARS_FORMAT": "png",  # Указываем формат наших exemplars
    }

    # Загружаем Recognizer с exemplars для one-shot detection
    print("Загружаем Recognizer из Logodetect...")
    recognizer = Recognizer(exemplars_path=str(exemplars_dir), config=config)
    
    # Восстанавливаем оригинальное значение BRAND_LOGOS
    constants.BRAND_LOGOS[:] = original_brand_logos

    return recognizer


def draw_bounding_boxes(
    frame_bgr: np.ndarray,
    recognition: Dict,
    recognizer,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Рисует bounding boxes на кадре на основе распознаваний от Logodetect.
    
    :param frame_bgr: кадр в формате BGR
    :param recognition: словарь с распознаваниями от Recognizer.compute_recognitions
    :param recognizer: экземпляр Recognizer для доступа к методу draw_overlay_boxes
    :param color: цвет для bounding box (не используется, т.к. используется метод recognizer)
    :param thickness: толщина линий (не используется, т.к. используется метод recognizer)
    :return: кадр с нарисованными bounding boxes
    """
    
    boxes = recognition.get("boxes", np.array([]))
    
    if len(boxes) == 0:
        return frame_bgr.copy()
    
    # Используем метод draw_overlay_boxes из Recognizer
    # Он ожидает RGB, поэтому конвертируем
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_with_boxes = recognizer.draw_overlay_boxes(frame_rgb, recognition)
    result_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
    
    return result_frame


def create_exemplar_folder(logo_path: Path, temp_dir: Path) -> Path:
    """Создает временную папку с exemplar для Logodetect."""
    
    exemplar_dir = temp_dir / "exemplar"
    exemplar_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем логотип в папку exemplar
    exemplar_file = exemplar_dir / logo_path.name
    shutil.copy2(logo_path, exemplar_file)
    
    return exemplar_dir


def detect_logos_on_frame(
    recognizer,
    frame_bgr: np.ndarray,
) -> Dict:
    """
    Детектирует логотипы на одном кадре используя Recognizer из Logodetect.
    
    :param recognizer: экземпляр Recognizer из Logodetect
    :param frame_bgr: кадр в формате BGR (OpenCV)
    :return: словарь с распознаваниями (boxes, labels, scores, brands)
    """
    
    # Конвертируем кадр в RGB для Logodetect
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Используем метод compute_recognitions из Recognizer
    # Он принимает numpy array (RGB) и возвращает список распознаваний
    # Внутри он передает numpy array в detector.predict, который может работать с numpy array
    recognitions = recognizer.compute_recognitions(frame_rgb, recognitions=[])
    
    if not recognitions:
        return {
            "boxes": np.array([]),
            "labels": np.array([]),
            "scores": np.array([]),
            "brands": [],
        }
    
    # Возвращаем первое (и единственное) распознавание
    return recognitions[0]


def process_video(
    video_path: Path,
    exemplar_path: Path,
    output_video: Path,
    intermediate_video: Path,
    recognizer,
    keep_intermediate: bool = False,
) -> Path:
    """Обрабатывает видео, детектируя логотипы на каждом кадре."""

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    exemplar_path = exemplar_path.resolve()
    if not exemplar_path.exists():
        raise FileNotFoundError(f"Exemplar логотипа не найден: {exemplar_path}")

    # Создаем временную папку для exemplar
    temp_dir = intermediate_video.parent / f"temp_{intermediate_video.stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        exemplar_dir = create_exemplar_folder(exemplar_path, temp_dir)

        # Читаем кадры видео
        frames_info = read_video_frames(video_path, seconds=None, frame_skip=1)
        height, width = frames_info.frame_size
        fps = frames_info.fps if frames_info.fps > 0 else 30.0

        intermediate_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(intermediate_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Не удалось открыть видеофайл для записи: {intermediate_video}")

        # Обрабатываем каждый кадр
        with tqdm(
            total=len(frames_info.frames_bgr),
            desc=video_path.name,
            unit="frame",
            dynamic_ncols=True,
        ) as progress:
            for frame_bgr in frames_info.frames_bgr:
                # Детектируем логотипы
                recognition = detect_logos_on_frame(recognizer, frame_bgr)
                
                # Рисуем bounding boxes
                frame_with_boxes = draw_bounding_boxes(frame_bgr, recognition, recognizer=recognizer)
                
                # Записываем кадр
                writer.write(frame_with_boxes)
                progress.update(1)

        writer.release()

        # Объединяем с аудио
        output_video.parent.mkdir(parents=True, exist_ok=True)
        mux_audio(intermediate_video, video_path, output_video)

        if not keep_intermediate and intermediate_video.exists():
            intermediate_video.unlink()
    finally:
        # Удаляем временную папку
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    return output_video


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


def load_logos_mapping(logos_list_path: Path) -> Dict[str, str]:
    """Загружает словарь соответствия видео и логотипов."""

    if not logos_list_path.exists():
        raise FileNotFoundError(f"Файл logos_list.json не найден: {logos_list_path}")

    data = json.loads(logos_list_path.read_text(encoding="utf-8"))
    return data


def main() -> int:
    """Главная функция."""

    parser = argparse.ArgumentParser(
        description="Детекция логотипов брендов на видео с помощью Logodetect"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Директория с видео",
    )
    parser.add_argument(
        "--logos-dir",
        type=Path,
        default=DEFAULT_LOGOS_DIR,
        help="Директория с логотипами",
    )
    parser.add_argument(
        "--logos-list",
        type=Path,
        default=DEFAULT_LOGOS_LIST,
        help="Путь к файлу logos_list.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Директория для выходных видео",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Обработать конкретное видео (по умолчанию обрабатываются все из logos_list.json)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Сохранять промежуточные видео без звука",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство для вычислений (cuda/cpu)",
    )

    args = parser.parse_args()

    # Клонируем и настраиваем Logodetect
    print("Настраиваем Logodetect...")
    repo_dir = ensure_logodetect_repo()
    install_logodetect_dependencies(repo_dir)

    # Загружаем соответствие видео и логотипов
    logos_mapping = load_logos_mapping(args.logos_list)
    
    # Определяем список видео для обработки
    if args.video:
        video_name = args.video.stem
        if video_name not in logos_mapping:
            print(f"Предупреждение: видео {video_name} не найдено в logos_list.json")
            return 1
        videos_to_process = {video_name: logos_mapping[video_name]}
    else:
        videos_to_process = logos_mapping

    # Обрабатываем каждое видео
    for video_name, logo_filename in videos_to_process.items():
        video_path = args.videos_dir / f"{video_name}.mp4"
        logo_path = args.logos_dir / logo_filename

        if not video_path.exists():
            print(f"Пропускаем {video_name}: видео не найдено")
            continue

        if not logo_path.exists():
            print(f"Пропускаем {video_name}: логотип не найден")
            continue

        # Создаем временную папку для exemplar этого видео
        temp_exemplar_dir = args.output_dir / f"temp_exemplar_{video_name}"
        temp_exemplar_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Копируем логотип в временную папку exemplar
            exemplar_file = temp_exemplar_dir / logo_path.name
            shutil.copy2(logo_path, exemplar_file)
            
            # Загружаем Recognizer с exemplar для этого видео
            device_str = args.device
            recognizer = load_logodetect_recognizer(repo_dir, temp_exemplar_dir, device_str, logo_filename)

            output_video = args.output_dir / f"{video_name}_logo_detection.mp4"
            intermediate_video = args.output_dir / f"{video_name}_logo_detection_silent.mp4"

            print(f"\nОбрабатываем {video_name}...")
            try:
                process_video(
                    video_path,
                    logo_path,
                    output_video,
                    intermediate_video,
                    recognizer,
                    keep_intermediate=args.keep_intermediate,
                )
                print(f"Готово: {output_video}")
            except Exception as e:
                print(f"Ошибка при обработке {video_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        finally:
            # Удаляем временную папку exemplar
            if temp_exemplar_dir.exists():
                shutil.rmtree(temp_exemplar_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

