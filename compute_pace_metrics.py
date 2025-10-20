"""Расчёт метрик темпа монтажа (pace / shot rate) для рекламных роликов."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

from pipeline_utils import as_posix, build_video_slug, find_video_file


DATASET_PATH = Path("VideoDataset.xlsx")
VIDEO_DIR = Path("data/videos")
OUTPUT_JSON = Path("data/analytics/pace_metrics.json")
OUTPUT_CSV = Path("data/analytics/pace_metrics.csv")


@dataclass
class PaceMetrics:
    """Метрики темпа монтажа для одного ролика."""

    duration_seconds: float
    n_cuts: int
    n_scenes: int
    shot_rate_per_sec: float
    median_shot_duration: float
    iqr_shot_duration: float
    cuts_in_first_5s: int
    first_window_seconds: float
    first_window_shot_rate: float


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _shot_durations(scene_boundaries: Iterable) -> List[float]:
    durations: List[float] = []
    for start_tc, end_tc in scene_boundaries:
        durations.append(max(end_tc.get_seconds() - start_tc.get_seconds(), 0.0))
    return durations


def analyse_video(
    video_path: Path,
    threshold: float,
    min_scene_len: int,
) -> PaceMetrics:
    video = open_video(video_path.as_posix())
    try:
        scene_manager = SceneManager()
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
        scene_manager.add_detector(detector)
        scene_manager.detect_scenes(video, show_progress=False)

        cut_list = scene_manager.get_cut_list()
        scene_list = scene_manager.get_scene_list()

        duration_tc = video.duration
        if duration_tc is None and hasattr(video, "get_duration"):
            duration_tc = video.get_duration()
        if duration_tc is None:
            raise RuntimeError("Не удалось определить длительность видео")
        duration_seconds = float(duration_tc.get_seconds())
        if duration_seconds <= 0:
            raise RuntimeError("Нулевая длительность видео")

        if not scene_list:
            scene_list = [(video.base_timecode, duration_tc)]

        durations = _shot_durations(scene_list)
        if not durations:
            durations = [duration_seconds]

        median_duration = float(np.median(durations))
        if len(durations) > 1:
            q75, q25 = np.percentile(durations, [75, 25])
            iqr = float(max(q75 - q25, 0.0))
        else:
            iqr = 0.0

        n_cuts = len(cut_list)
        n_scenes = len(scene_list)
        shot_rate = float(n_cuts) / duration_seconds

        window = min(duration_seconds, 5.0)
        if window <= 0:
            window = duration_seconds
        cuts_first_window = sum(1 for cut in cut_list if cut.get_seconds() <= 5.0)
        first_window_rate = (cuts_first_window / window) if window > 0 else 0.0

        return PaceMetrics(
            duration_seconds=duration_seconds,
            n_cuts=n_cuts,
            n_scenes=n_scenes,
            shot_rate_per_sec=shot_rate,
            median_shot_duration=median_duration,
            iqr_shot_duration=iqr,
            cuts_in_first_5s=cuts_first_window,
            first_window_seconds=window,
            first_window_shot_rate=first_window_rate,
        )
    finally:
        if hasattr(video, "close"):
            video.close()
        elif hasattr(video, "release"):
            video.release()


def process_dataset(
    dataset_path: Path,
    video_dir: Path,
    threshold: float,
    min_scene_len: int,
) -> List[Dict[str, object]]:
    df = pd.read_excel(dataset_path)
    results: List[Dict[str, object]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Pace metrics"):
        slug = build_video_slug(idx, row.get("name"), row.get("link"))
        video_path = find_video_file(video_dir, slug)
        entry: Dict[str, object] = {
            "index": int(idx),
            "slug": slug,
            "video_path": as_posix(video_path),
            "name": row.get("name"),
            "link": row.get("link"),
        }

        if video_path is None:
            entry["status"] = "missing_video"
            results.append(entry)
            continue

        try:
            metrics = analyse_video(video_path, threshold=threshold, min_scene_len=min_scene_len)
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
        else:
            entry["status"] = "ok"
            entry.update(asdict(metrics))
        results.append(entry)

    return results


def save_outputs(
    entries: List[Dict[str, object]],
    output_json: Path,
    output_csv: Optional[Path],
    dataset_path: Path,
    video_dir: Path,
    threshold: float,
    min_scene_len: int,
) -> None:
    ensure_parent(output_json)
    payload = {
        "dataset": dataset_path.as_posix(),
        "video_dir": as_posix(video_dir),
        "detector": {
            "type": "content",
            "threshold": threshold,
            "min_scene_len": min_scene_len,
        },
        "entries": entries,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if output_csv is not None:
        ensure_parent(output_csv)
        rows = [entry for entry in entries if entry.get("status") == "ok"]
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv, index=False)
        else:
            output_csv.write_text("", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Рассчитать Pace / Shot Rate для видеороликов")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Путь к таблице с метаданными")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR, help="Каталог с видеороликами")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help="Путь для сохранения JSON с результатами",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Путь для сохранения CSV (только успешные записи)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Порог ContentDetector в PySceneDetect",
    )
    parser.add_argument(
        "--min-scene-len",
        type=int,
        default=15,
        help="Минимальная длина сцены в кадрах для ContentDetector",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Не сохранять CSV-файл",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = process_dataset(
        dataset_path=args.dataset,
        video_dir=args.video_dir,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
    )
    output_csv = None if args.no_csv else args.output_csv
    save_outputs(
        entries=entries,
        output_json=args.output_json,
        output_csv=output_csv,
        dataset_path=args.dataset,
        video_dir=args.video_dir,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
    )


if __name__ == "__main__":
    main()

