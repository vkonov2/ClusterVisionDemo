"""Visualize PySceneDetect ContentDetector metrics for a single video."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


class ContentDebugError(RuntimeError):
    """Raised when ContentDetector analysis cannot be completed."""


@dataclass
class SceneSummary:
    index: int
    start_timecode: str
    end_timecode: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass
class AnalysisArtifacts:
    output_dir: Path
    metrics_csv: Path
    plot_path: Path | None
    frames_dir: Path | None
    summary_json: Path


def _import_pyscenedetect():
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
        from scenedetect.stats_manager import StatsManager
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ContentDebugError(
            "PySceneDetect is required to analyze ContentDetector output."
        ) from exc
    return SceneManager, open_video, ContentDetector, StatsManager


def _plot_metrics(
    metrics_csv: Path,
    threshold: float,
    cut_times: Sequence[float],
    output_path: Path,
    dpi: int = 150,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ContentDebugError(
            "matplotlib is required to generate ContentDetector visualizations."
        ) from exc

    frame = pd.read_csv(metrics_csv)
    if "Timecode" not in frame.columns:
        raise ContentDebugError(
            "Metrics CSV does not contain a 'Timecode' column."
        )

    seconds = pd.to_timedelta(frame["Timecode"]).dt.total_seconds()
    if "content_val" not in frame.columns:
        raise ContentDebugError(
            "Metrics CSV does not contain the 'content_val' score produced by ContentDetector."
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(seconds, frame["content_val"], label="content_val", color="#1f77b4")
    axes[0].axhline(threshold, color="#d62728", linestyle="--", label="threshold")
    for cut in cut_times:
        axes[0].axvline(cut, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Frame score")
    axes[0].set_title("ContentDetector frame score vs. threshold")
    axes[0].legend()

    component_cols = [
        col
        for col in ("delta_hue", "delta_sat", "delta_lum", "delta_edges")
        if col in frame.columns
    ]
    for col in component_cols:
        axes[1].plot(seconds, frame[col], label=col)
    for cut in cut_times:
        axes[1].axvline(cut, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Component score")
    if component_cols:
        axes[1].set_title("ContentDetector component deltas")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _export_cut_frames(
    video_path: Path,
    cut_frames: Sequence[int],
    output_dir: Path,
    frame_radius: int,
) -> None:
    if frame_radius < 0:
        return

    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ContentDebugError(
            "opencv-python is required to export cut preview frames."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise ContentDebugError(f"Unable to open video with OpenCV: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        for idx, frame_number in enumerate(cut_frames):
            offsets = range(-frame_radius, frame_radius + 1)
            for offset in offsets:
                target_frame = frame_number + offset
                if target_frame < 0 or (
                    total_frames > 0 and target_frame >= total_frames
                ):
                    continue

                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue

                suffix = "center" if offset == 0 else ("m%02d" % abs(offset) if offset < 0 else "p%02d" % offset)
                filename = f"cut_{idx:03d}_{suffix}.jpg"
                cv2.imwrite(output_dir.joinpath(filename).as_posix(), frame)
    finally:
        capture.release()


def analyze_content_detector(
    video_path: Path,
    output_root: Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    downscale_factor: int = 1,
    frame_radius: int = 1,
    skip_frames: bool = False,
    show_progress: bool = False,
    dpi: int = 150,
) -> AnalysisArtifacts:
    SceneManager, open_video, ContentDetector, StatsManager = _import_pyscenedetect()

    if not video_path.exists():
        raise ContentDebugError(f"Video file does not exist: {video_path}")
    if downscale_factor < 1:
        raise ContentDebugError("downscale_factor must be >= 1")
    if frame_radius < 0:
        raise ContentDebugError("frame_radius must be >= 0")

    video = open_video(video_path.as_posix(), framerate=None)
    if downscale_factor > 1:
        video.downscale = downscale_factor

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager=stats_manager)
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    scene_manager.add_detector(detector)

    try:
        scene_manager.detect_scenes(video, show_progress=show_progress)
        scene_list = scene_manager.get_scene_list()
        if not scene_list:
            raise ContentDebugError("ContentDetector did not find any scenes.")

        cut_list = scene_manager.get_cut_list()
        cut_seconds = [cut.get_seconds() for cut in cut_list]
        cut_frames = [cut.get_frames() for cut in cut_list]

        target_dir = output_root / video_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        metrics_csv = target_dir / "content_metrics.csv"
        stats_manager.save_to_csv(metrics_csv)

        plot_path: Path | None = target_dir / "content_detector_plot.png"
        _plot_metrics(metrics_csv, threshold, cut_seconds, plot_path, dpi=dpi)

        frames_dir: Path | None = None
        if not skip_frames and cut_frames:
            frames_dir = target_dir / "frames"
            _export_cut_frames(video_path, cut_frames, frames_dir, frame_radius)

        summary = [
            SceneSummary(
                index=i,
                start_timecode=scene[0].get_timecode(),
                end_timecode=scene[1].get_timecode(),
                start_seconds=scene[0].get_seconds(),
                end_seconds=scene[1].get_seconds(),
                duration_seconds=(scene[1] - scene[0]).get_seconds(),
            )
            for i, scene in enumerate(scene_list)
        ]

        duration_attr = getattr(video, "duration", None)
        if duration_attr is None:
            duration_seconds = None
        elif hasattr(duration_attr, "get_seconds"):
            duration_seconds = duration_attr.get_seconds()
        else:
            duration_seconds = float(duration_attr)

        summary_payload = {
            "video_path": video_path.as_posix(),
            "frame_rate": getattr(video, "frame_rate", None),
            "frame_size": getattr(video, "frame_size", None),
            "duration_seconds": duration_seconds,
            "threshold": threshold,
            "min_scene_len": min_scene_len,
            "downscale_factor": downscale_factor,
            "scene_count": len(scene_list),
            "cut_count": len(cut_list),
            "scenes": [asdict(item) for item in summary],
        }

        summary_json = target_dir / "scene_summary.json"
        summary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        close = getattr(video, "close", None)
        if callable(close):
            close()
        else:
            release = getattr(video, "release", None)
            if callable(release):
                release()

    return AnalysisArtifacts(
        output_dir=target_dir,
        metrics_csv=metrics_csv,
        plot_path=plot_path,
        frames_dir=frames_dir,
        summary_json=summary_json,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a single video with PySceneDetect's ContentDetector and generate visualizations.",
    )
    parser.add_argument("video", type=Path, help="Path to the video file to analyze")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analytics/shot_debug"),
        help="Directory where analysis artifacts will be stored",
    )
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=15)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument(
        "--frame-radius",
        type=int,
        default=1,
        help="Number of frames before/after each cut to export for visual inspection",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Disable exporting preview frames around detected cuts",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress output during scene detection",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Resolution of the generated plot")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        artifacts = analyze_content_detector(
            video_path=args.video,
            output_root=args.output_dir,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            downscale_factor=args.downscale_factor,
            frame_radius=args.frame_radius,
            skip_frames=args.skip_frames,
            show_progress=not args.no_progress,
            dpi=args.dpi,
        )
    except ContentDebugError as exc:
        parser.error(str(exc))
        return 2

    print(f"Saved ContentDetector metrics to {artifacts.metrics_csv}")
    if artifacts.plot_path:
        print(f"Saved ContentDetector visualization to {artifacts.plot_path}")
    if artifacts.frames_dir:
        print(f"Exported preview frames to {artifacts.frames_dir}")
    print(f"Saved scene summary JSON to {artifacts.summary_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
