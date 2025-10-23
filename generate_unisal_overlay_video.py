"""Render a UNISAL saliency overlay for every frame and mux it with the source audio."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from generate_unisal_saliency import (
    ensure_unisal_repo,
    load_unisal_model,
    read_video_frames,
    build_preprocess_transform,
    preprocess_frames,
    run_unisal_inference,
    render_heatmap,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/unisal")
DEFAULT_INPUT_DIR = Path("data/videos")
DEFAULT_GLOB_PATTERN = "*.mp4"
DEFAULT_WORKER_COUNT = max(1, (os.cpu_count() or 1) - 2)


@dataclass(frozen=True)
class VideoJob:
    video_path: Path
    output_video: Path
    intermediate_video: Path
    seconds: float | None
    chunk_size: int
    alpha: float
    source: str
    keep_intermediate: bool
    progress_position: int = 0


_WORKER_MODEL: "torch.nn.Module" | None = None
_WORKER_DEVICE: torch.device | None = None
_WORKER_GET_OPTIMAL_OUT_SIZE: Callable[[Tuple[int, int]], Tuple[int, int]] | None = None


def _worker_initializer(repo_dir_str: str, device_spec: str) -> None:
    """Load the UNISAL model once per worker process."""

    repo_dir = ensure_unisal_repo(Path(repo_dir_str))
    sys.path.insert(0, str(repo_dir))
    from unisal.data import get_optimal_out_size  # type: ignore

    device = torch.device(device_spec)
    model = load_unisal_model(repo_dir, device)

    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_GET_OPTIMAL_OUT_SIZE
    _WORKER_MODEL = model
    _WORKER_DEVICE = device
    _WORKER_GET_OPTIMAL_OUT_SIZE = get_optimal_out_size


def _ensure_worker_ready() -> tuple["torch.nn.Module", torch.device, Callable[[Tuple[int, int]], Tuple[int, int]]]:
    if _WORKER_MODEL is None or _WORKER_DEVICE is None or _WORKER_GET_OPTIMAL_OUT_SIZE is None:
        raise RuntimeError("Worker model has not been initialised")
    return _WORKER_MODEL, _WORKER_DEVICE, _WORKER_GET_OPTIMAL_OUT_SIZE


def chunk_sequence(sequence: Sequence[np.ndarray], chunk_size: int) -> Iterator[Sequence[np.ndarray]]:
    """Yield non-overlapping chunks from the provided sequence."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    for start in range(0, len(sequence), chunk_size):
        yield sequence[start : start + chunk_size]


def overlay_frame(frame_bgr: np.ndarray, prob_map: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the rendered saliency map with the original frame."""

    heatmap = render_heatmap(prob_map)
    beta = 1.0 - alpha
    return cv2.addWeighted(frame_bgr, beta, heatmap, alpha, 0)


def write_overlay_video(
    frames: Sequence[np.ndarray],
    prob_maps_iter: Iterable[np.ndarray],
    output_path: Path,
    fps: float,
    alpha: float,
    progress: "tqdm" | None = None,
) -> None:
    """Write overlay frames to an mp4 container."""

    if not frames:
        raise ValueError("No frames available for writing")

    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")

    iterator = iter(prob_maps_iter)
    for index, frame in enumerate(frames):
        try:
            prob_map = next(iterator)
        except StopIteration as exc:
            writer.release()
            raise RuntimeError(
                "Model produced fewer saliency maps than frames processed"
            ) from exc
        overlay = overlay_frame(frame, prob_map, alpha)
        writer.write(overlay)
        if progress is not None:
            progress.update(1)

    try:
        extra = next(iterator)
    except StopIteration:
        extra = None

    writer.release()

    if extra is not None:
        raise RuntimeError("Model produced more saliency maps than frames processed")


def process_video(
    video_path: Path,
    output_video: Path,
    intermediate_video: Path,
    *,
    seconds: float | None,
    chunk_size: int,
    alpha: float,
    device: torch.device,
    model: "torch.nn.Module",
    source: str,
    keep_intermediate: bool,
    get_optimal_out_size: Callable[[Tuple[int, int]], Tuple[int, int]],
    progress_position: int = 0,
) -> Path:
    frames_info = read_video_frames(video_path, seconds, frame_skip=1)
    height, width = frames_info.frame_size
    fps = frames_info.fps if frames_info.fps > 0 else 30.0

    out_size = get_optimal_out_size((height, width))
    transform = build_preprocess_transform(out_size)

    prob_maps_generator = iter_prob_maps(
        frames_info.frames_bgr,
        max(1, chunk_size),
        transform,
        model,
        device,
        target_size=(height, width),
        source=source,
    )

    intermediate_video.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(frames_info.frames_bgr),
        desc=video_path.name,
        unit="frame",
        position=progress_position,
        leave=True,
        dynamic_ncols=True,
    ) as frame_bar:
        write_overlay_video(
            frames_info.frames_bgr,
            prob_maps_generator,
            intermediate_video,
            fps,
            alpha=alpha,
            progress=frame_bar,
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    mux_audio(intermediate_video, video_path, output_video)

    if not keep_intermediate and intermediate_video.exists():
        intermediate_video.unlink()

    frame_bar.write(f"Overlay video with audio written to {output_video}")
    return output_video


def mux_audio(
    silent_video: Path,
    source_video: Path,
    output_video: Path,
    reencode: bool = True,
) -> None:
    """Mux the overlay video with the original audio track using ffmpeg."""

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
            "ffmpeg failed to mux audio and video:\n"
            f"STDOUT: {result.stdout.decode('utf-8', errors='ignore')}\n"
            f"STDERR: {result.stderr.decode('utf-8', errors='ignore')}"
        )


def iter_prob_maps(
    frames: Sequence[np.ndarray],
    chunk_size: int,
    transform,
    model: "torch.nn.Module",
    device: torch.device,
    target_size: tuple[int, int],
    source: str,
) -> Iterator[np.ndarray]:
    """Yield probability maps for the provided frames."""

    for chunk in chunk_sequence(frames, chunk_size):
        video_tensor = preprocess_frames(chunk, transform)
        predictions = run_unisal_inference(
            model,
            video_tensor,
            device,
            target_size=target_size,
            source=source,
        )
        prob_maps = predictions.exp().squeeze(0).squeeze(1).numpy()
        for prob_map in prob_maps:
            yield prob_map


def process_video_job(job: VideoJob) -> Path:
    model, device, get_optimal_out_size = _ensure_worker_ready()
    return process_video(
        job.video_path,
        job.output_video,
        job.intermediate_video,
        seconds=job.seconds,
        chunk_size=job.chunk_size,
        alpha=job.alpha,
        device=device,
        model=model,
        source=job.source,
        keep_intermediate=job.keep_intermediate,
        get_optimal_out_size=get_optimal_out_size,
        progress_position=job.progress_position,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render UNISAL saliency overlays for every frame and mux with original audio",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Process only this video path (skips directory traversal)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing videos to process when --video is omitted",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help="Glob pattern for selecting videos inside --input-dir",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated overlay videos",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Destination for the muxed overlay video (requires --video)",
    )
    parser.add_argument(
        "--intermediate-video",
        type=Path,
        default=None,
        help="Temporary silent overlay video path (requires --video)",
    )
    parser.add_argument("--seconds", type=float, default=None, help="Limit processing to the first N seconds")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of frames to process per model forward pass",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay strength applied to the saliency heatmap",
    )
    parser.add_argument("--device", type=str, default=None, help="Force computation device (cpu or cuda)")
    parser.add_argument(
        "--source",
        type=str,
        default="DHF1K",
        help="UNISAL domain to use for BatchNorm statistics",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the silent overlay video instead of deleting it",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help="Number of parallel worker processes to use (CPU only)",
    )
    args = parser.parse_args()

    if args.video is None and (args.output_video is not None or args.intermediate_video is not None):
        parser.error("--output-video and --intermediate-video require --video to be set")

    repo_dir = ensure_unisal_repo()
    device_spec = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_spec)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > 1 and device.type == "cuda":
        parser.error("--workers > 1 is not supported when using CUDA devices")

    sys.path.insert(0, str(repo_dir))
    from unisal.data import get_optimal_out_size  # type: ignore

    output_root = args.output_root.expanduser()
    explicit_output = args.output_video.expanduser() if args.output_video else None
    explicit_intermediate = args.intermediate_video.expanduser() if args.intermediate_video else None

    videos_to_process: List[Path]
    single_video_resolved: Path | None = None
    if args.video is not None:
        video_path = args.video.expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        videos_to_process = [video_path]
        single_video_resolved = video_path.resolve()
    else:
        input_dir = args.input_dir.expanduser()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        videos_to_process = sorted(
            path for path in input_dir.glob(args.pattern) if path.is_file()
        )
        if not videos_to_process:
            raise FileNotFoundError(
                f"No videos matching pattern '{args.pattern}' found in {input_dir}"
            )

    jobs: List[VideoJob] = []
    for video_path in videos_to_process:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        resolved_video = video_path.resolve()
        is_explicit_target = (
            single_video_resolved is not None and resolved_video == single_video_resolved
        )

        if is_explicit_target and explicit_output is not None:
            output_video = explicit_output
        else:
            output_video = output_root / video_path.stem / "overlay_with_audio.mp4"

        if is_explicit_target and explicit_intermediate is not None:
            intermediate_video = explicit_intermediate
        else:
            intermediate_video = output_video.with_name("overlay_video_silent.mp4")

        jobs.append(
            VideoJob(
                video_path=resolved_video,
                output_video=output_video,
                intermediate_video=intermediate_video,
                seconds=args.seconds,
                chunk_size=args.chunk_size,
                alpha=args.alpha,
                source=args.source,
                keep_intermediate=args.keep_intermediate,
            )
        )

    if args.workers == 1:
        model = load_unisal_model(repo_dir, device)
        with tqdm(jobs, desc="Videos", unit="video", position=0) as video_bar:
            for job in video_bar:
                process_video(
                    job.video_path,
                    job.output_video,
                    job.intermediate_video,
                    seconds=job.seconds,
                    chunk_size=job.chunk_size,
                    alpha=job.alpha,
                    device=device,
                    model=model,
                    source=job.source,
                    keep_intermediate=job.keep_intermediate,
                    get_optimal_out_size=get_optimal_out_size,
                    progress_position=1,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_initializer,
            initargs=(str(repo_dir), device_spec),
        ) as executor:
            pending: set = set()
            future_to_position: dict = {}
            available_positions = deque(range(args.workers))
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
                        future.result()
                    except Exception:
                        for remaining in pending:
                            if remaining is not future:
                                remaining.cancel()
                        raise
                    available_positions.append(position)

                pending = not_done
                try:
                    while available_positions:
                        job = next(jobs_iter)
                        submit_job(job, available_positions.popleft())
                except StopIteration:
                    continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
