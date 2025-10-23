"""Render a UNISAL saliency overlay for every frame and mux it with the source audio."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import cv2
import numpy as np
import torch

from generate_unisal_saliency import (
    DEFAULT_VIDEO,
    ensure_unisal_repo,
    load_unisal_model,
    read_video_frames,
    build_preprocess_transform,
    preprocess_frames,
    run_unisal_inference,
    render_heatmap,
)


DEFAULT_OUTPUT_VIDEO = Path("outputs/unisal/000-youtube/overlay_with_audio.mp4")
DEFAULT_INTERMEDIATE_VIDEO = Path("outputs/unisal/000-youtube/overlay_video_silent.mp4")


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

    try:
        extra = next(iterator)
    except StopIteration:
        extra = None

    writer.release()

    if extra is not None:
        raise RuntimeError("Model produced more saliency maps than frames processed")


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render UNISAL saliency overlays for every frame and mux with original audio",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument(
        "--intermediate-video",
        type=Path,
        default=DEFAULT_INTERMEDIATE_VIDEO,
        help="Temporary silent overlay video path",
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
    args = parser.parse_args()

    repo_dir = ensure_unisal_repo()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_unisal_model(repo_dir, device)

    frames_info = read_video_frames(args.video, args.seconds, frame_skip=1)
    height, width = frames_info.frame_size
    fps = frames_info.fps if frames_info.fps > 0 else 30.0

    sys.path.insert(0, str(repo_dir))
    from unisal.data import get_optimal_out_size  # type: ignore

    out_size = get_optimal_out_size((height, width))
    transform = build_preprocess_transform(out_size)

    prob_maps_generator = iter_prob_maps(
        frames_info.frames_bgr,
        max(1, args.chunk_size),
        transform,
        model,
        device,
        target_size=(height, width),
        source=args.source,
    )

    write_overlay_video(
        frames_info.frames_bgr,
        prob_maps_generator,
        args.intermediate_video,
        fps,
        alpha=args.alpha,
    )

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    mux_audio(args.intermediate_video, args.video, args.output_video)

    if not args.keep_intermediate and args.intermediate_video.exists():
        args.intermediate_video.unlink()

    print(f"Overlay video with audio written to {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
