"""Generate saliency heatmaps for a video using the UNISAL model."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.",
    category=UserWarning,
)

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


REPO_URL = "https://github.com/rdroste/unisal.git"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "unisal"
REPO_DIR = CACHE_DIR / "repo"
DEFAULT_VIDEO = Path("data/videos/000-youtube.mp4")
DEFAULT_OUTPUT = Path("outputs/unisal/000-youtube")


@dataclass
class VideoFrames:
    frames_bgr: List[np.ndarray]
    fps: float

    @property
    def frame_size(self) -> Tuple[int, int]:
        if not self.frames_bgr:
            raise ValueError("No frames captured from video")
        h, w = self.frames_bgr[0].shape[:2]
        return h, w


def ensure_unisal_repo(repo_dir: Path = REPO_DIR, repo_url: str = REPO_URL) -> Path:
    """Clone the UNISAL repository into the cache directory if needed."""

    repo_dir = repo_dir.expanduser().resolve()
    if repo_dir.exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning UNISAL repository to {repo_dir}...")
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    return repo_dir


def load_unisal_model(repo_dir: Path, device: torch.device) -> "torch.nn.Module":
    """Load the pretrained UNISAL model."""

    sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("TRAIN_DIR", str(repo_dir / "training_runs"))
    os.environ.setdefault("PRED_DIR", str(repo_dir / "predictions"))

    import unisal  # type: ignore  # noqa: WPS433 - third-party module vendored at runtime

    model = unisal.model.UNISAL()
    weights_dir = repo_dir / "training_runs" / "pretrained_unisal"
    if not weights_dir.exists():
        raise FileNotFoundError(f"Pretrained weights not found in {weights_dir}")

    model.load_best_weights(weights_dir)
    model.to(device)
    model.eval()
    return model


def read_video_frames(video_path: Path, seconds: float | None, frame_skip: int) -> VideoFrames:
    """Read video frames up to the requested duration."""

    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = total_frames
    if seconds is not None and fps > 0:
        max_frames = min(total_frames, int(fps * seconds))

    collected: List[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break
        if frame_index % frame_skip == 0:
            collected.append(frame_bgr)
        frame_index += 1
        if frame_index >= max_frames:
            break

    cap.release()
    if not collected:
        raise RuntimeError("No frames were read from the video")

    return VideoFrames(frames_bgr=collected, fps=fps)


def build_preprocess_transform(out_size: Tuple[int, int]) -> transforms.Compose:
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(out_size, interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ]
    )


def preprocess_frames(frames: Sequence[np.ndarray], transform: transforms.Compose) -> torch.Tensor:
    tensors = [transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    stacked = torch.stack(tensors)  # [time, channel, height, width]
    return stacked.unsqueeze(0)  # [batch, time, channel, height, width]


def run_unisal_inference(
    model: "torch.nn.Module",
    video_tensor: torch.Tensor,
    device: torch.device,
    target_size: Tuple[int, int],
    source: str,
) -> torch.Tensor:
    with torch.no_grad():
        predictions = model(
            video_tensor.to(device),
            target_size=target_size,
            source=source,
            static=False,
        )
    return predictions.cpu()


def prepare_output_dir(path: Path) -> Path:
    """Create a clean output directory for the current run."""

    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_heatmap(prob_map: np.ndarray) -> np.ndarray:
    if prob_map.max() <= 0:
        normalized = np.zeros_like(prob_map, dtype=np.uint8)
    else:
        normalized = (prob_map / prob_map.max() * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def save_frame_outputs(
    output_dir: Path,
    frame_idx: int,
    frame_bgr: np.ndarray,
    prob_map: np.ndarray,
) -> None:
    heatmap = render_heatmap(prob_map)
    overlay = cv2.addWeighted(frame_bgr, 0.55, heatmap, 0.45, 0)

    heatmap_path = output_dir / f"frame_{frame_idx:04d}_heatmap.png"
    overlay_path = output_dir / f"frame_{frame_idx:04d}_overlay.png"
    cv2.imwrite(str(heatmap_path), heatmap)
    cv2.imwrite(str(overlay_path), overlay)


def save_summary_outputs(
    output_dir: Path,
    frames: Sequence[np.ndarray],
    prob_maps: np.ndarray,
    metadata: dict,
) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    mean_map = prob_maps.mean(axis=0)
    max_map = prob_maps.max(axis=0)

    first_frame = frames[0]
    for name, map_ in {"mean": mean_map, "max": max_map}.items():
        heatmap = render_heatmap(map_)
        overlay = cv2.addWeighted(first_frame, 0.55, heatmap, 0.45, 0)
        cv2.imwrite(str(summary_dir / f"{name}_heatmap.png"), heatmap)
        cv2.imwrite(str(summary_dir / f"{name}_overlay.png"), overlay)

    (summary_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_metadata(
    video_path: Path,
    repo_dir: Path,
    device: torch.device,
    frames: Sequence[np.ndarray],
    prob_maps: np.ndarray,
    fps: float,
    seconds: float | None,
    frame_skip: int,
    sample_step: int,
    source: str,
) -> dict:
    hook_strength = float(np.mean(prob_maps.max(axis=(1, 2))))
    return {
        "video_path": str(video_path),
        "frames_processed": len(frames),
        "fps": fps,
        "seconds_limit": seconds,
        "frame_skip": frame_skip,
        "sample_step": sample_step,
        "model_repo": REPO_URL,
        "model_cache": str(repo_dir),
        "device": str(device),
        "source_domain": source,
        "hook_strength": hook_strength,
        "mean_map_peak": float(prob_maps.mean(axis=0).max()),
        "max_map_peak": float(prob_maps.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UNISAL saliency inference on a video")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=5.0, help="Duration of the clip to analyse")
    parser.add_argument("--frame-skip", type=int, default=1, help="Sample every N-th frame from the source video")
    parser.add_argument(
        "--sample-step",
        type=int,
        default=3,
        help="Store overlays for every N-th processed frame",
    )
    parser.add_argument("--max-saved-frames", type=int, default=6, help="Maximum number of per-frame overlays to save")
    parser.add_argument("--source", type=str, default="DHF1K", help="UNISAL domain to use for BatchNorm statistics")
    parser.add_argument("--device", type=str, default=None, help="Force computation device (cpu or cuda)")
    args = parser.parse_args()

    repo_dir = ensure_unisal_repo()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_unisal_model(repo_dir, device)

    frames_info = read_video_frames(args.video, args.seconds, max(1, args.frame_skip))
    h, w = frames_info.frame_size

    sys.path.insert(0, str(repo_dir))
    from unisal.data import get_optimal_out_size  # type: ignore

    out_size = get_optimal_out_size((h, w))
    transform = build_preprocess_transform(out_size)
    video_tensor = preprocess_frames(frames_info.frames_bgr, transform)

    predictions = run_unisal_inference(
        model,
        video_tensor,
        device,
        target_size=(h, w),
        source=args.source,
    )

    prob_maps = predictions.exp().squeeze(0).squeeze(1).numpy()

    output_dir = prepare_output_dir(args.output)
    saved = 0
    for idx, (frame, prob_map) in enumerate(zip(frames_info.frames_bgr, prob_maps)):
        if args.sample_step <= 0 or idx % args.sample_step == 0:
            save_frame_outputs(output_dir, idx, frame, prob_map)
            saved += 1
        if saved >= args.max_saved_frames:
            break

    metadata = compute_metadata(
        args.video,
        repo_dir,
        device,
        frames_info.frames_bgr,
        prob_maps,
        frames_info.fps,
        args.seconds,
        args.frame_skip,
        args.sample_step,
        args.source,
    )
    save_summary_outputs(output_dir, frames_info.frames_bgr, prob_maps, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
