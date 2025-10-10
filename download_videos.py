"""Скачивание рекламных роликов из VideoDataset.xlsx."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from pipeline_utils import as_posix, build_video_slug, find_video_file


DATASET_PATH = Path("VideoDataset.xlsx")
OUTPUT_DIR = Path("data/videos")
MANIFEST_PATH = OUTPUT_DIR / "download_manifest.json"

MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 65.0
MAX_FILESIZE_BYTES = 200 * 1024 * 1024


def iter_candidate_sources(row: pd.Series) -> Iterable[str]:
    sources: List[str] = []
    seen: set[str] = set()

    def yield_candidate(text: str) -> None:
        lower = text.lower()
        if lower.startswith("http"):
            sources.append(text)
        elif "youtube" in lower or "youtu.be" in lower:
            sources.append(text)
            sources.append(f"ytsearch1:{text}")
        else:
            sources.append(f"ytsearch1:{text}")

    def push(value: Optional[str]) -> None:
        if not isinstance(value, str):
            return
        candidate = value.strip()
        if not candidate or candidate.lower() == "nan":
            return
        if candidate in seen:
            return
        seen.add(candidate)
        yield_candidate(candidate)

    push(row.get("link"))
    push(row.get("mirror"))
    push(row.get("name"))
    push(row.get("extra"))
    if not sources:
        sources.append(f"ytsearch1:video {row.name}")
    return sources


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_video_info(info: Dict) -> Dict:
    if not isinstance(info, dict):
        return {}
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        for entry in entries:
            if isinstance(entry, dict):
                return entry
        return {}
    return info


def _estimate_filesize_bytes(info: Dict) -> Optional[int]:
    for key in ("filesize", "filesize_approx"):
        size = info.get(key)
        if isinstance(size, (int, float)) and size > 0:
            return int(size)

    requested_formats = info.get("requested_formats")
    if isinstance(requested_formats, list) and requested_formats:
        total = 0
        found = False
        for fmt in requested_formats:
            if not isinstance(fmt, dict):
                continue
            for key in ("filesize", "filesize_approx"):
                size = fmt.get(key)
                if isinstance(size, (int, float)) and size > 0:
                    total += int(size)
                    found = True
                    break
        if found:
            return total

    size = info.get("filesize") or info.get("filesize_approx")
    if isinstance(size, (int, float)) and size > 0:
        return int(size)

    return None


def _validate_constraints(info: Dict) -> Tuple[bool, Optional[str]]:
    duration = info.get("duration")
    if not isinstance(duration, (int, float)):
        return False, "duration_unknown"
    if duration <= MIN_DURATION_SECONDS:
        return False, "duration_too_short"
    if duration >= MAX_DURATION_SECONDS:
        return False, "duration_too_long"

    filesize = _estimate_filesize_bytes(info)
    if filesize is None:
        return False, "filesize_unknown"
    if filesize > MAX_FILESIZE_BYTES:
        return False, "filesize_too_large"

    return True, None


def download_single_video(slug: str, sources: Iterable[str], output_dir: Path) -> Dict[str, object]:
    outtmpl = (output_dir / f"{slug}.%(ext)s").as_posix()
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b/best",
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        "noplaylist": True,
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
        "default_search": "ytsearch",
    }

    last_error: Optional[str] = None
    for source in sources:
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                if info is None:
                    last_error = f"Empty info for {source}"
                    continue
                video_info = _resolve_video_info(info)
                if not video_info:
                    last_error = f"Unresolved info for {source}"
                    continue

                is_valid, reason = _validate_constraints(video_info)
                if not is_valid:
                    return {
                        "status": "skipped",
                        "reason": reason,
                        "source": source,
                        "title": video_info.get("title"),
                        "duration": video_info.get("duration"),
                        "filepath": None,
                    }

                download_code = ydl.download([source])
                if download_code != 0:
                    last_error = f"yt-dlp exited with code {download_code}"
                    continue

                filename = Path(ydl.prepare_filename(video_info))
                if not filename.exists():
                    # yt-dlp мог конвертировать в mp4, проверим с поиском
                    existing = find_video_file(output_dir, slug)
                    if existing is not None:
                        filename = existing

                if not filename.exists():
                    last_error = f"Downloaded file not found for {source}"
                    continue

                file_size = filename.stat().st_size
                if file_size > MAX_FILESIZE_BYTES:
                    filename.unlink(missing_ok=True)
                    return {
                        "status": "skipped",
                        "reason": "filesize_too_large",
                        "source": source,
                        "title": video_info.get("title"),
                        "duration": video_info.get("duration"),
                        "filepath": None,
                    }

                return {
                    "status": "downloaded",
                    "source": source,
                    "title": video_info.get("title"),
                    "duration": video_info.get("duration"),
                    "filepath": as_posix(filename),
                }
        except DownloadError as exc:
            message = str(exc)
            if "sign in to confirm your age" in message.lower():
                return {
                    "status": "skipped",
                    "reason": "age_restricted",
                    "source": source,
                    "title": None,
                    "duration": None,
                    "filepath": None,
                }
            last_error = message
            continue
        except Exception as exc:  # yt-dlp raises различное
            last_error = str(exc)
            continue

    return {"status": "failed", "error": last_error or "unknown", "source": None, "filepath": None}


def main(dataset: Path = DATASET_PATH, output_dir: Path = OUTPUT_DIR, manifest_path: Path = MANIFEST_PATH) -> int:
    ensure_output_dir(output_dir)

    df = pd.read_excel(dataset)
    entries: List[Dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Download videos"):
        slug = build_video_slug(idx, row.get("name"), row.get("link"))
        record: Dict[str, object] = {
            "index": int(idx),
            "slug": slug,
            "link": row.get("link"),
            "mirror": row.get("mirror"),
            "name": row.get("name"),
        }

        existing_path = find_video_file(output_dir, slug)

        if existing_path is not None:
            record.update({
                "status": "exists",
                "filepath": as_posix(existing_path),
                "source": None,
            })
            entries.append(record)
            continue

        sources = iter_candidate_sources(row)
        result = download_single_video(slug, sources, output_dir)
        if result.get("status") == "downloaded":
            final_path = find_video_file(output_dir, slug)
            if final_path is not None:
                result["filepath"] = as_posix(final_path)
        record.update(result)
        entries.append(record)

    manifest = {
        "dataset": as_posix(dataset),
        "output_dir": as_posix(output_dir),
        "entries": entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}")

    failed = sum(1 for e in entries if e.get("status") == "failed")
    skipped = sum(1 for e in entries if e.get("status") == "skipped")
    print(
        "Downloaded {downloaded} videos, failed: {failed}, skipped: {skipped}".format(
            downloaded=len(entries) - failed - skipped,
            failed=failed,
            skipped=skipped,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos listed in VideoDataset.xlsx")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    sys.exit(main(args.dataset, args.output_dir, args.manifest))

