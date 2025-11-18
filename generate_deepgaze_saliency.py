"""Генерация тепловых карт внимания для видео с помощью моделей DeepGaze."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from generate_unisal_saliency import read_video_frames, render_heatmap


REPO_URL = "https://github.com/matthias-k/DeepGaze.git"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "deepgaze"
REPO_DIR = CACHE_DIR / "repo"
CENTERBIAS_URL = (
    "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
)
CENTERBIAS_PATH = CACHE_DIR / "centerbias_mit1003.npy"

DEFAULT_VIDEO = Path("data/videos/000-youtube.mp4")
DEFAULT_OUTPUT = Path("outputs/deepgaze/000-youtube")
DEFAULT_MODEL = "deepgaze2e"


@dataclass
class DeepGazePrediction:
    frame_index: int
    frame_bgr: np.ndarray
    prob_map: np.ndarray


def ensure_deepgaze_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Клонировать репозиторий DeepGaze при первом запуске."""

    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Клонируем DeepGaze в {repo_dir}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def ensure_centerbias(path: Path = CENTERBIAS_PATH) -> Path:
    """Загрузить лог-плотность центр-биаса MIT1003."""

    path = path.expanduser().resolve()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Загружаем центр-биас MIT1003...")
    with urllib.request.urlopen(CENTERBIAS_URL) as response:
        data = response.read()
    path.write_bytes(data)
    return path


def load_deepgaze_model(repo_dir: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    """Подгрузить предобученную модель DeepGaze."""

    sys.path.insert(0, str(repo_dir))
    import deepgaze_pytorch  # type: ignore  # noqa: WPS433

    if model_name == "deepgaze2e":
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
    elif model_name == "deepgaze3":
        model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
    else:
        raise ValueError(f"Неизвестная модель DeepGaze: {model_name}")

    model.to(device)
    model.eval()
    return model


def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1)))
    return tensor.float()


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


def log_density_to_prob(log_density: np.ndarray) -> np.ndarray:
    stable = log_density - log_density.max()
    prob = np.exp(stable)
    total = prob.sum()
    if total <= 0:
        return np.zeros_like(prob)
    return prob / total


def init_scanpath_history(width: int, height: int, model: torch.nn.Module) -> deque[tuple[float, float]]:
    included = getattr(model, "included_fixations", [])
    if not included:
        return deque()
    history_len = max(abs(index) for index in included)
    center = (width / 2.0, height / 2.0)
    return deque([center for _ in range(history_len)], maxlen=history_len)


def prepare_history_tensors(
    history: Sequence[tuple[float, float]],
    included: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not included:
        raise ValueError("DeepGaze III требует историю фиксаций")

    history_array = np.array(history, dtype=np.float32)
    x_hist = torch.from_numpy(np.ascontiguousarray(history_array[included, 0])).unsqueeze(0)
    y_hist = torch.from_numpy(np.ascontiguousarray(history_array[included, 1])).unsqueeze(0)
    return x_hist.to(device), y_hist.to(device)


def iterate_deepgaze_predictions(
    frames: Sequence[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    centerbias_log: np.ndarray,
    *,
    chunk_size: int,
) -> Iterable[DeepGazePrediction]:
    height, width = frames[0].shape[:2]
    centerbias_tensor = torch.from_numpy(centerbias_log).float().to(device)

    if isinstance(model, torch.nn.Module) and getattr(model, "included_fixations", []):
        history = init_scanpath_history(width, height, model)
        included = list(getattr(model, "included_fixations"))
        if not history:
            raise RuntimeError("Не удалось инициализировать историю для DeepGaze III")

        for idx, frame in enumerate(frames):
            frame_tensor = frame_to_tensor(frame).unsqueeze(0).to(device)
            x_hist, y_hist = prepare_history_tensors(history, included, device)
            with torch.no_grad():
                log_density = model(frame_tensor, centerbias_tensor.unsqueeze(0), x_hist, y_hist)
            log_map = log_density.squeeze().detach().cpu().numpy()
            prob_map = log_density_to_prob(log_map)
            max_pos = np.unravel_index(np.argmax(prob_map), prob_map.shape)
            history.append((float(max_pos[1]), float(max_pos[0])))
            yield DeepGazePrediction(idx, frame, prob_map)
        return

    batch_tensors: list[torch.Tensor] = []
    batch_frames: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        batch_tensors.append(frame_to_tensor(frame))
        batch_frames.append(frame)
        if len(batch_tensors) < max(1, chunk_size):
            continue
        yield from _run_batch(
            batch_tensors,
            batch_frames,
            idx - len(batch_tensors) + 1,
            model,
            device,
            centerbias_tensor,
        )
        batch_tensors = []
        batch_frames = []

    if batch_tensors:
        yield from _run_batch(
            batch_tensors,
            batch_frames,
            len(frames) - len(batch_tensors),
            model,
            device,
            centerbias_tensor,
        )


def _run_batch(
    tensors: list[torch.Tensor],
    frames: list[np.ndarray],
    start_index: int,
    model: torch.nn.Module,
    device: torch.device,
    centerbias_tensor: torch.Tensor,
) -> Iterable[DeepGazePrediction]:
    images = torch.stack(tensors).to(device)
    cb = centerbias_tensor.unsqueeze(0).expand(images.size(0), -1, -1)
    with torch.no_grad():
        log_density = model(images, cb)
    maps = log_density.squeeze(1).detach().cpu().numpy()
    for offset, log_map in enumerate(maps):
        prob_map = log_density_to_prob(log_map)
        yield DeepGazePrediction(start_index + offset, frames[offset], prob_map)


def prepare_output_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions(
    predictions: Iterable[DeepGazePrediction],
    output_dir: Path,
    sample_step: int,
    max_saved: int,
) -> list[np.ndarray]:
    saved = 0
    all_prob_maps: list[np.ndarray] = []
    for pred in predictions:
        all_prob_maps.append(pred.prob_map)
        if saved < max_saved and (sample_step <= 0 or pred.frame_index % sample_step == 0):
            heatmap = render_heatmap(pred.prob_map)
            overlay = cv2.addWeighted(pred.frame_bgr, 0.55, heatmap, 0.45, 0)
            cv2.imwrite(str(output_dir / f"frame_{pred.frame_index:04d}_heatmap.png"), heatmap)
            cv2.imwrite(str(output_dir / f"frame_{pred.frame_index:04d}_overlay.png"), overlay)
            saved += 1
    return all_prob_maps


def compute_metadata(
    video_path: Path,
    repo_dir: Path,
    device: torch.device,
    frames_count: int,
    prob_maps: Sequence[np.ndarray],
    fps: float,
    seconds: float | None,
    frame_skip: int,
    sample_step: int,
    model_name: str,
    uniform_centerbias: bool,
    centerbias_path: Path | None,
) -> dict:
    hook_strength = float(np.mean([prob.max() for prob in prob_maps])) if prob_maps else 0.0
    return {
        "video_path": str(video_path),
        "frames_processed": frames_count,
        "fps": fps,
        "seconds_limit": seconds,
        "frame_skip": frame_skip,
        "sample_step": sample_step,
        "model_repo": REPO_URL,
        "model_cache": str(repo_dir),
        "device": str(device),
        "model_variant": model_name,
        "centerbias": "uniform" if uniform_centerbias else str(centerbias_path),
        "hook_strength": hook_strength,
        "mean_map_peak": float(np.mean(prob_maps, axis=0).max()) if prob_maps else 0.0,
        "max_map_peak": float(np.max(prob_maps)) if prob_maps else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Инференс DeepGaze по видео")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", choices=["deepgaze2e", "deepgaze3"], default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=5.0, help="Длительность анализируемого фрагмента")
    parser.add_argument("--frame-skip", type=int, default=1, help="Использовать каждый N-й кадр")
    parser.add_argument("--sample-step", type=int, default=3, help="Сохранять оверлеи через N кадров")
    parser.add_argument("--max-saved-frames", type=int, default=6, help="Максимум сохраняемых оверлеев")
    parser.add_argument("--chunk-size", type=int, default=4, help="Размер батча для DeepGaze2E")
    parser.add_argument("--centerbias", type=Path, default=None, help="Файл с лог-плотностью центр-биаса")
    parser.add_argument("--uniform-centerbias", action="store_true", help="Использовать равномерный центр-биас")
    parser.add_argument("--device", type=str, default=None, help="Указать устройство (cpu или cuda)")
    args = parser.parse_args()

    repo_dir = ensure_deepgaze_repo()
    if args.uniform_centerbias:
        centerbias_template = None
        centerbias_path = None
    else:
        centerbias_path = args.centerbias or ensure_centerbias()
        centerbias_template = np.load(centerbias_path)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_deepgaze_model(repo_dir, args.model, device)

    frames_info = read_video_frames(args.video, args.seconds, max(1, args.frame_skip))
    height, width = frames_info.frame_size
    centerbias_log = build_centerbias(
        height,
        width,
        uniform=args.uniform_centerbias,
        template=centerbias_template,
    )

    predictions = list(
        iterate_deepgaze_predictions(
            frames_info.frames_bgr,
            model,
            device,
            centerbias_log,
            chunk_size=max(1, args.chunk_size),
        )
    )

    output_dir = prepare_output_dir(args.output)
    prob_maps = save_predictions(
        predictions,
        output_dir,
        args.sample_step,
        args.max_saved_frames,
    )

    mean_map = np.mean(prob_maps, axis=0) if prob_maps else np.zeros_like(centerbias_log)
    max_map = np.max(prob_maps, axis=0) if prob_maps else np.zeros_like(centerbias_log)

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    first_frame = predictions[0].frame_bgr if predictions else frames_info.frames_bgr[0]
    for name, map_ in {"mean": mean_map, "max": max_map}.items():
        heatmap = render_heatmap(map_)
        overlay = cv2.addWeighted(first_frame, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(str(summary_dir / f"{name}_heatmap.png"), heatmap)
        cv2.imwrite(str(summary_dir / f"{name}_overlay.png"), overlay)

    metadata = compute_metadata(
        args.video,
        repo_dir,
        device,
        len(frames_info.frames_bgr),
        prob_maps,
        frames_info.fps,
        args.seconds,
        args.frame_skip,
        args.sample_step,
        args.model,
        args.uniform_centerbias,
        centerbias_path,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
