"""Утилиты для пайплайна обработки рекламных роликов."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional


SAFE_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".m4v",
    ".avi",
    ".mpg",
    ".mpeg",
)


def _normalize_to_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text


def slugify_text(text: str) -> str:
    ascii_text = _normalize_to_ascii(text).lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-")


def build_video_slug(index: int, *candidates: Iterable[str]) -> str:
    """Возвращает стабильный slug для строки датасета."""

    flattened = []
    for candidate in candidates:
        if isinstance(candidate, str) and candidate and candidate == candidate:
            flattened.append(candidate)
        elif isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
            for item in candidate:
                if isinstance(item, str) and item and item == item:
                    flattened.append(item)

    for text in flattened:
        slug = slugify_text(text)
        if slug:
            return f"{index:03d}-{slug[:64]}"

    digest = hashlib.sha1("|".join(flattened).encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{index:03d}-video-{digest[:10]}"


def find_video_file(video_dir: Path, slug: str) -> Optional[Path]:
    """Возвращает путь к скачанному видео по slug (если существует)."""

    for ext in SAFE_VIDEO_EXTENSIONS:
        candidate = video_dir / f"{slug}{ext}"
        if candidate.exists():
            return candidate

    matches = sorted(video_dir.glob(f"{slug}.*"))
    for match in matches:
        if match.suffix.lower() in SAFE_VIDEO_EXTENSIONS:
            return match
    return matches[0] if matches else None


def as_posix(path: Optional[Path]) -> Optional[str]:
    return path.as_posix() if path is not None else None

