"""Построение видео с тепловыми картами внимания от моделей DeepGaze."""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from generate_deepgaze_saliency import (
    build_centerbias,
    ensure_centerbias,
    ensure_deepgaze_repo,
    frame_to_tensor,
    init_scanpath_history,
    load_deepgaze_model,
    log_density_to_prob,
    prepare_history_tensors,
)
from generate_unisal_saliency import read_video_frames, render_heatmap


DEFAULT_OUTPUT_ROOT = Path("outputs/deepgaze")
DEFAULT_INPUT_DIR = Path("data/videos")
DEFAULT_GLOB_PATTERN = "*.mp4"
DEFAULT_WORKER_COUNT = 1


@dataclass(frozen=True)
class VideoJob:
    video_path: Path
    output_video: Path
    intermediate_video: Path
    seconds: float | None
    chunk_size: int
    alpha: float
    keep_intermediate: bool
    progress_position: int = 0


_WORKER_MODEL: torch.nn.Module | None = None
_WORKER_DEVICE: torch.device | None = None
_WORKER_MODEL_NAME: str | None = None
_WORKER_UNIFORM_CENTERBIAS: bool = False
_WORKER_CENTERBIAS_TEMPLATE: np.ndarray | None = None


def _worker_initializer(
    repo_dir_str: str,
    device_spec: str,
    model_name: str,
    centerbias_path_str: str,
    uniform_centerbias: bool,
) -> None:
    repo_dir = ensure_deepgaze_repo(Path(repo_dir_str))
    sys.path.insert(0, str(repo_dir))

    device = torch.device(device_spec)
    model = load_deepgaze_model(repo_dir, model_name, device)

    template: np.ndarray | None = None
    if not uniform_centerbias:
        template_path = Path(centerbias_path_str) if centerbias_path_str else ensure_centerbias()
        template = np.load(template_path)

    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_MODEL_NAME
    global _WORKER_UNIFORM_CENTERBIAS, _WORKER_CENTERBIAS_TEMPLATE
    _WORKER_MODEL = model
    _WORKER_DEVICE = device
    _WORKER_MODEL_NAME = model_name
    _WORKER_UNIFORM_CENTERBIAS = uniform_centerbias
    _WORKER_CENTERBIAS_TEMPLATE = template


def _ensure_worker_ready() -> tuple[torch.nn.Module, torch.device, str, bool, np.ndarray | None]:
    if _WORKER_MODEL is None or _WORKER_DEVICE is None or _WORKER_MODEL_NAME is None:
        raise RuntimeError("DeepGaze модель ещё не загружена в воркере")
    return (
        _WORKER_MODEL,
        _WORKER_DEVICE,
        _WORKER_MODEL_NAME,
        _WORKER_UNIFORM_CENTERBIAS,
        _WORKER_CENTERBIAS_TEMPLATE,
    )


def overlay_frame(frame_bgr: np.ndarray, prob_map: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = render_heatmap(prob_map)
    beta = 1.0 - alpha
    return cv2.addWeighted(frame_bgr, beta, heatmap, alpha, 0)


def write_overlay_video(
    frames: Sequence[np.ndarray],
    prob_maps_iter: Iterable[np.ndarray],
    output_path: Path,
    fps: float,
    alpha: float,
    progress: tqdm | None = None,
) -> None:
    if not frames:
        raise ValueError("Нет кадров для записи")

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть видеофайл для записи: {output_path}")

    iterator = iter(prob_maps_iter)
    for frame in frames:
        try:
            prob_map = next(iterator)
        except StopIteration as exc:
            writer.release()
            raise RuntimeError("Модель вернула меньше карт, чем кадров") from exc
        writer.write(overlay_frame(frame, prob_map, alpha))
        if progress is not None:
            progress.update(1)

    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        writer.release()
        raise RuntimeError("Модель вернула больше карт, чем кадров")

    writer.release()


def iterate_prob_maps(
    frames: Sequence[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    centerbias_tensor: torch.Tensor,
    centerbias_log: np.ndarray,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    if getattr(model, "included_fixations", []) and model_name.lower().endswith("3"):
        history = init_scanpath_history(centerbias_log.shape[1], centerbias_log.shape[0], model)
        included = list(getattr(model, "included_fixations"))
        for frame in frames:
            frame_tensor = frame_to_tensor(frame).unsqueeze(0).to(device)
            x_hist, y_hist = prepare_history_tensors(history, included, device)
            with torch.no_grad():
                log_density = model(frame_tensor, centerbias_tensor.unsqueeze(0), x_hist, y_hist)
            log_map = log_density.squeeze().detach().cpu().numpy()
            prob_map = log_density_to_prob(log_map)
            max_pos = np.unravel_index(np.argmax(prob_map), prob_map.shape)
            history.append((float(max_pos[1]), float(max_pos[0])))
            yield prob_map
        return

    batch_tensors: list[torch.Tensor] = []
    for frame in frames:
        batch_tensors.append(frame_to_tensor(frame))
        if len(batch_tensors) < max(1, chunk_size):
            continue
        yield from _run_batch_prob(batch_tensors, model, device, centerbias_tensor)
        batch_tensors = []

    if batch_tensors:
        yield from _run_batch_prob(batch_tensors, model, device, centerbias_tensor)


def _run_batch_prob(
    tensors: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    centerbias_tensor: torch.Tensor,
) -> Iterator[np.ndarray]:
    images = torch.stack(tensors).to(device)
    cb = centerbias_tensor.unsqueeze(0).expand(images.size(0), -1, -1)
    with torch.no_grad():
        log_density = model(images, cb)
    maps = log_density.squeeze(1).detach().cpu().numpy()
    for log_map in maps:
        yield log_density_to_prob(log_map)


def process_video(
    video_path: Path,
    output_video: Path,
    intermediate_video: Path,
    *,
    seconds: float | None,
    chunk_size: int,
    alpha: float,
    keep_intermediate: bool,
    progress_position: int = 0,
) -> Path:
    model, device, model_name, uniform_centerbias, centerbias_template = _ensure_worker_ready()

    frames_info = read_video_frames(video_path, seconds, frame_skip=1)
    height, width = frames_info.frame_size
    fps = frames_info.fps if frames_info.fps > 0 else 30.0

    centerbias_log = build_centerbias(
        height,
        width,
        uniform=uniform_centerbias,
        template=centerbias_template,
    )
    centerbias_tensor = torch.from_numpy(centerbias_log).float().to(device)

    if model_name.lower().endswith("3"):
        chunk_size = 1

    with tqdm(
        total=len(frames_info.frames_bgr),
        desc=video_path.name,
        unit="frame",
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
    ) as frame_bar:
        prob_maps = iterate_prob_maps(
            frames_info.frames_bgr,
            model,
            device,
            model_name,
            centerbias_tensor,
            centerbias_log,
            max(1, chunk_size),
        )
        write_overlay_video(
            frames_info.frames_bgr,
            prob_maps,
            intermediate_video,
            fps,
            alpha=alpha,
            progress=frame_bar,
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    mux_audio(intermediate_video, video_path, output_video)

    if not keep_intermediate and intermediate_video.exists():
        intermediate_video.unlink()

    return output_video


def mux_audio(
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    reencode: bool = True,
) -> None:
    cmd: List[str] = [
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
            "ffmpeg не смог объединить видео и аудио:\n"
            f"STDOUT: {result.stdout.decode('utf-8', errors='ignore')}\n"
            f"STDERR: {result.stderr.decode('utf-8', errors='ignore')}"
        )


def process_video_job(job: VideoJob) -> Path:
    return process_video(
        job.video_path,
        job.output_video,
        job.intermediate_video,
        seconds=job.seconds,
        chunk_size=job.chunk_size,
        alpha=job.alpha,
        keep_intermediate=job.keep_intermediate,
        progress_position=job.progress_position,
    )


def prepare_jobs(
    videos: Sequence[Path],
    output_root: Path,
    *,
    output_video: Path | None,
    intermediate_video: Path | None,
    seconds: float | None,
    chunk_size: int,
    alpha: float,
    keep_intermediate: bool,
    model_name: str,
) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for index, video_path in enumerate(videos):
        if output_video is not None and intermediate_video is not None and len(videos) == 1:
            final_path = output_video
            silent_path = intermediate_video
        else:
            rel_dir = video_path.stem
            final_path = output_root / rel_dir / f"{video_path.stem}_{model_name}_overlay.mp4"
            silent_path = output_root / rel_dir / f"{video_path.stem}_{model_name}_silent.mp4"
        jobs.append(
            VideoJob(
                video_path=video_path,
                output_video=final_path,
                intermediate_video=silent_path,
                seconds=seconds,
                chunk_size=chunk_size,
                alpha=alpha,
                keep_intermediate=keep_intermediate,
                progress_position=index,
            )
        )
    return jobs


def gather_videos(input_dir: Path, pattern: str) -> list[Path]:
    videos = sorted(input_dir.glob(pattern))
    return [path for path in videos if path.is_file()]


def schedule_jobs(
    jobs: Sequence[VideoJob],
    *,
    workers: int,
    initializer,
    init_args: tuple,
) -> list[Path]:
    if workers <= 1:
        results: list[Path] = []
        for job in jobs:
            results.append(process_video_job(replace(job, progress_position=0)))
        return results

    completed: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=init_args,
    ) as executor:
        pending: set = set()
        future_to_position: dict = {}
        available_positions = deque(range(workers))
        jobs_iter = iter(jobs)

        def submit_job(job: VideoJob, position: int) -> None:
            job_with_position = replace(job, progress_position=position)
            future = executor.submit(process_video_job, job_with_position)
            pending.add(future)
            future_to_position[future] = position

        try:
            while available_positions:
                job = next(jobs_iter)
                submit_job(job, available_positions.popleft())
        except StopIteration:
            pass

        while pending:
            done, not_done = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                position = future_to_position.pop(future)
                try:
                    result_path = future.result()
                except Exception:
                    for remaining in pending:
                        if remaining is not future:
                            remaining.cancel()
                    raise
                completed.append(result_path)
                available_positions.append(position)
                pending.remove(future)

            pending = not_done
            try:
                while available_positions:
                    job = next(jobs_iter)
                    submit_job(job, available_positions.popleft())
            except StopIteration:
                continue

    return completed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Построить видео с картами внимания DeepGaze и вернуть звук из оригинала",
    )
    parser.add_argument("--video", type=Path, default=None, help="Обработать только указанный файл")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Папка с видео при пакетной обработке",
    )
    parser.add_argument("--pattern", type=str, default=DEFAULT_GLOB_PATTERN, help="Глоб паттерн для поиска видео")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Корневая папка для результатов",
    )
    parser.add_argument("--output-video", type=Path, default=None, help="Выходной файл (только с --video)")
    parser.add_argument("--intermediate-video", type=Path, default=None, help="Временный файл без аудио")
    parser.add_argument("--seconds", type=float, default=None, help="Ограничить обработку первыми N секундами")
    parser.add_argument("--chunk-size", type=int, default=4, help="Размер батча для DeepGaze IIe")
    parser.add_argument("--alpha", type=float, default=0.45, help="Прозрачность тепловой карты")
    parser.add_argument("--keep-intermediate", action="store_true", help="Не удалять временное видео без аудио")
    parser.add_argument("--device", type=str, default=None, help="Указать устройство (cpu/cuda)")
    parser.add_argument("--model", choices=["deepgaze2e", "deepgaze3"], default="deepgaze2e")
    parser.add_argument("--centerbias", type=Path, default=None, help="Файл лог-плотности центр-биаса")
    parser.add_argument("--uniform-centerbias", action="store_true", help="Использовать равномерный центр-биас")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT, help="Количество параллельных воркеров")
    parser.add_argument("--overwrite", action="store_true", help="Перезаписать существующие результаты")
    args = parser.parse_args()

    if args.video is not None and (args.output_video is None or args.intermediate_video is None):
        if args.output_video is None or args.intermediate_video is None:
            raise SystemExit("Для одиночного видео нужно указать --output-video и --intermediate-video")

    videos: list[Path]
    if args.video is not None:
        if not args.video.exists():
            raise SystemExit(f"Видео не найдено: {args.video}")
        videos = [args.video]
    else:
        if not args.input_dir.exists():
            raise SystemExit(f"Папка не найдена: {args.input_dir}")
        videos = gather_videos(args.input_dir, args.pattern)
    if not videos:
        print("Видео для обработки не найдены")
        return 0

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = prepare_jobs(
        videos,
        output_root,
        output_video=args.output_video,
        intermediate_video=args.intermediate_video,
        seconds=args.seconds,
        chunk_size=max(1, args.chunk_size),
        alpha=args.alpha,
        keep_intermediate=args.keep_intermediate,
        model_name=args.model,
    )

    if not args.overwrite:
        filtered_jobs: list[VideoJob] = []
        for job in jobs:
            if job.output_video.exists():
                print(f"Пропускаю {job.video_path.name}: результат уже существует")
                continue
            filtered_jobs.append(job)
        jobs = filtered_jobs
    if not jobs:
        print("Все видео уже обработаны")
        return 0

    repo_dir = ensure_deepgaze_repo()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.uniform_centerbias:
        centerbias_path = None
    else:
        centerbias_path = args.centerbias or ensure_centerbias()

    init_args = (
        str(repo_dir),
        str(device),
        args.model,
        str(centerbias_path) if centerbias_path else "",
        args.uniform_centerbias,
    )

    results = schedule_jobs(
        [replace(job, chunk_size=max(1, args.chunk_size)) for job in jobs],
        workers=max(1, args.workers),
        initializer=_worker_initializer,
        init_args=init_args,
    )

    for path in results:
        print(f"Готово: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
