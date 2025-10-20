"""Compute shot pacing metrics for all downloaded videos using PySceneDetect."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_utils import as_posix, build_video_slug, find_video_file


@dataclass
class ShotMetrics:
    duration: float
    scene_count: int
    cut_count: int
    cuts_per_second: float
    median_shot_duration: float
    iqr_shot_duration: float
    cuts_in_first_5s: int
    cuts_in_first_5s_per_second: float


class SceneDetectionError(RuntimeError):
    """Raised when scene detection cannot be performed."""


def _frame_time_to_seconds(frame_time) -> float:
    if frame_time is None:
        return 0.0
    if hasattr(frame_time, "get_seconds"):
        return float(frame_time.get_seconds())
    if hasattr(frame_time, "to_seconds"):
        return float(frame_time.to_seconds())
    if hasattr(frame_time, "seconds"):
        return float(frame_time.seconds)
    return float(frame_time)


def _extract_scene_bounds(scene) -> tuple[float, float]:
    if scene is None:
        return 0.0, 0.0
    if hasattr(scene, "start_time") and hasattr(scene, "end_time"):
        start = scene.start_time
        end = scene.end_time
    elif hasattr(scene, "start") and hasattr(scene, "end"):
        start = scene.start
        end = scene.end
    elif isinstance(scene, Iterable):
        # PySceneDetect<=0.5 returns tuples of FrameTimecode
        items = list(scene)
        if len(items) >= 2:
            start, end = items[0], items[1]
        elif len(items) == 1:
            start, end = items[0], items[0]
        else:
            start = end = 0.0
    else:
        start = end = scene
    return _frame_time_to_seconds(start), _frame_time_to_seconds(end)


def detect_shots(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    downscale_factor: int = 1,
) -> ShotMetrics:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SceneDetectionError(
            "PySceneDetect is required to compute shot metrics."
        ) from exc

    if not video_path.exists():
        raise SceneDetectionError(f"Video file does not exist: {video_path}")

    video = open_video(video_path.as_posix(), framerate=None)
    if downscale_factor > 1:
        video.downscale = downscale_factor

    try:
        scene_manager = SceneManager()
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        scene_manager.add_detector(detector)
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
    finally:
        video.close()

    if not scene_list:
        raise SceneDetectionError("No scenes were detected.")

    durations: List[float] = []
    cut_times: List[float] = []

    for idx, scene in enumerate(scene_list):
        start_s, end_s = _extract_scene_bounds(scene)
        durations.append(max(end_s - start_s, 0.0))
        if idx < len(scene_list) - 1:
            cut_times.append(end_s)

    video_duration = float(_extract_scene_bounds(scene_list[-1])[1])
    cut_count = len(cut_times)
    scene_count = len(scene_list)

    cuts_per_second = cut_count / video_duration if video_duration > 0 else 0.0
    cuts_in_first_5s = sum(1 for ts in cut_times if ts <= 5.0)
    first_window = min(video_duration, 5.0)
    cuts_in_first_5s_per_second = (
        cuts_in_first_5s / first_window if first_window > 0 else 0.0
    )

    if durations:
        median_shot_duration = float(statistics.median(durations))
        if len(durations) >= 2:
            q1, q3 = np.percentile(durations, [25, 75])
            iqr_shot_duration = float(q3 - q1)
        else:
            iqr_shot_duration = 0.0
    else:
        median_shot_duration = 0.0
        iqr_shot_duration = 0.0

    return ShotMetrics(
        duration=video_duration,
        scene_count=scene_count,
        cut_count=cut_count,
        cuts_per_second=cuts_per_second,
        median_shot_duration=median_shot_duration,
        iqr_shot_duration=iqr_shot_duration,
        cuts_in_first_5s=cuts_in_first_5s,
        cuts_in_first_5s_per_second=cuts_in_first_5s_per_second,
    )


def compute_metrics_for_dataset(
    dataset_path: Path,
    video_dir: Path,
    threshold: float,
    min_scene_len: int,
    downscale_factor: int,
) -> pd.DataFrame:
    df = pd.read_excel(dataset_path)
    records: List[dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyze videos"):
        slug = build_video_slug(idx, row.get("name"), row.get("link"))
        record = {
            "index": int(idx),
            "slug": slug,
            "name": row.get("name"),
            "link": row.get("link"),
        }

        video_path = find_video_file(video_dir, slug)
        if video_path is None:
            record.update({"status": "missing", "filepath": None})
            records.append(record)
            continue

        record["filepath"] = as_posix(video_path)

        try:
            metrics = detect_shots(
                video_path,
                threshold=threshold,
                min_scene_len=min_scene_len,
                downscale_factor=downscale_factor,
            )
        except SceneDetectionError as exc:
            record.update({"status": "error", "error": str(exc)})
            records.append(record)
            continue

        record.update(
            {
                "status": "ok",
                "video_duration": metrics.duration,
                "scene_count": metrics.scene_count,
                "cut_count": metrics.cut_count,
                "cuts_per_second": metrics.cuts_per_second,
                "median_shot_duration": metrics.median_shot_duration,
                "iqr_shot_duration": metrics.iqr_shot_duration,
                "cuts_in_first_5s": metrics.cuts_in_first_5s,
                "cuts_in_first_5s_per_second": metrics.cuts_in_first_5s_per_second,
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute shot pacing metrics for videos")
    parser.add_argument("--dataset", type=Path, default=Path("VideoDataset.xlsx"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/videos"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analytics/shot_metrics.csv"),
    )
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min-scene-len", type=int, default=15)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--write-json", action="store_true", help="Save metrics as JSON alongside CSV")
    args = parser.parse_args()

    if args.downscale_factor < 1:
        parser.error("--downscale-factor must be >= 1")

    frame = compute_metrics_for_dataset(
        dataset_path=args.dataset,
        video_dir=args.video_dir,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
        downscale_factor=args.downscale_factor,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved metrics to {args.output}")

    if args.write_json:
        json_path = args.output.with_suffix(".json")
        json_path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON to {json_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
