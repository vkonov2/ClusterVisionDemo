#!/usr/bin/env python3
"""Generate text-only captions for short videos using the UGC-VideoCaptioner model.

Features
- Downloads source video from MinIO when an HTTP/S3 URL is provided.
- Writes the generated caption to JSON and optionally uploads it back to MinIO.
- Updates Postgres job/video records in the same way as other analytics tools in the repo.
- Does **not** synthesize or return any audio narration to keep runtime lean.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv
import torch

# Root paths and environment loading -----------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

ENV_FILES = [REPO_ROOT / ".env", REPO_ROOT / ".env.local", APP_DIR / ".env.local-video2text"]
for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file, override=False)

# Database models from the main API package
from api.app.models.artifact import Artifact  # type: ignore
from api.app.models.job import Job, JobStatus  # type: ignore
from api.app.models.user import User  # type: ignore  # noqa: F401
from api.app.models.video import Video, VideoStatus  # type: ignore

import boto3
from botocore.client import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@dataclass
class CaptionResult:
    caption: str
    video_path: Path
    frames_used: int
    model_id: str


# Utility helpers ------------------------------------------------------------------------------
def parse_minio_path(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https", "s3"}:
        path = parsed.path.lstrip("/")
        if not path or "/" not in path:
            raise ValueError("URL must contain bucket and key, e.g. http://host/bucket/key")
        bucket, key = path.split("/", 1)
        return bucket, key
    raise ValueError("Unsupported URL: provide http(s) or s3 URL pointing to MinIO")


def build_minio_client(endpoint: str | None, access_key: str | None, secret_key: str | None):
    if not endpoint or not access_key or not secret_key:
        raise ValueError("MINIO endpoint/access/secret are required for remote operations")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )


def download_minio_object(client, uri: str, destination_dir: Path) -> Path:
    bucket, key = parse_minio_path(uri)
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_path = destination_dir / Path(key).name
    client.download_file(bucket, key, str(local_path))
    return local_path


def register_artifact(db, *, video_id: str | None, bucket: str, key: str, artifact_type: str):
    if db is None or video_id is None:
        return
    uri = f"s3://{bucket}/{key}"
    artifact = Artifact(video_id=video_id, type=artifact_type, uri=uri)
    db.add(artifact)
    db.commit()


def update_statuses(
    db,
    *,
    job,
    video,
    job_status: JobStatus | None = None,
    video_status: VideoStatus | None = None,
    progress: int | None = None,
    error: str | None = None,
):
    if not db:
        return

    changed = False
    if job is not None and job_status is not None:
        job.status = job_status
        changed = True
    if job is not None and progress is not None:
        job.progress = progress
        changed = True
    if job is not None and error is not None:
        job.error = error
        changed = True
    if video is not None and video_status is not None:
        video.status = video_status
        changed = True
    if changed:
        db.commit()


def build_db_sessionmaker(database_url: str | None):
    if not database_url:
        return None
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# Video caption generation ---------------------------------------------------------------------
def build_prompt() -> str:
    lines = (
        "You are a video-to-text transcription model. Describe the video with maximal detail and complete neutrality.",
        "Do NOT infer or guess intent, emotions, or target audience — only describe what is observable.",
        "Your output must follow this structure:",
        "1. Characters and Demographics: Number of people, their approximate age range, gender presentation, clothing style, notable visual traits.",
        "2. Setting and Environment: Indoor/outdoor location, objects present, visual cues.",
        "3. Narrative Summary: Step-by-step description of what happens in the video.",
        "4. Product and Branding: Any products, packaging, or logos shown.",
        "5. Text and Speech: All spoken lines and on-screen text.",
        "6. Visual and Symbolic Details: Colors, lighting, props, symbols, recognizable items.",
        "7. Audio and Music: Genre, tempo, instruments, sound effects, voice-over characteristics.",
        "8. Visual Style and Editing: Camera movements, shot duration, transitions, pacing, framing style.",
        "9. Call to Action: Any explicit CTA shown or spoken.",
        "Describe everything fully, factually, and concretely, without adding analysis or opinions.",
    )
    return "\n".join(lines)


def load_video_frames(video_path: Path, target_fps: float = 2.0, max_frames: int = 256):
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    native_fps = float(meta.get("fps", target_fps) or target_fps)
    fps = target_fps if native_fps <= 0 else native_fps
    step = max(int(round(fps / target_fps)), 1)

    frames: List[Image.Image] = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frames.append(Image.fromarray(frame).convert("RGB"))
        if len(frames) >= max_frames:
            break
    reader.close()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")
    return frames


def generate_caption(
    *,
    video_path: Path,
    device: str | None = None,
    max_new_tokens: int = 320,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> CaptionResult:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    prompt_text = build_prompt()
    frames = load_video_frames(video_path)

    model_id = "openinterx/UGC-VideoCaptioner"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        dtype="auto",
        device_map={"": device or ("cuda" if torch.cuda.is_available() else "cpu")},
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 2.0},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    use_audio_in_video = False
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    generated_tokens, _ = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    captions = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    caption_text = "\n".join([caption.strip() for caption in captions])
    return CaptionResult(
        caption=caption_text,
        video_path=video_path,
        frames_used=len(frames),
        model_id=model_id,
    )


async def run_workflow(args: argparse.Namespace) -> dict[str, Any]:
    minio_client = None
    need_minio = bool(args.video_url or not args.skip_minio_upload)
    if need_minio:
        minio_client = build_minio_client(args.minio_endpoint, args.minio_access_key, args.minio_secret_key)

    SessionLocal = build_db_sessionmaker(args.database_url)
    db = SessionLocal() if SessionLocal and (args.job_id or args.video_id) else None
    job = db.get(Job, args.job_id) if db and args.job_id else None
    video = db.get(Video, args.video_id) if db and args.video_id else None

    update_statuses(
        db,
        job=job,
        video=video,
        job_status=JobStatus.STARTED,
        video_status=VideoStatus.PROCESSING,
        progress=0,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            video_path = args.video
            if args.video_url:
                if not minio_client:
                    raise ValueError("MinIO client is required to download video")
                video_path = download_minio_object(minio_client, args.video_url, tmp_path)

            result = await asyncio.to_thread(
                generate_caption,
                video_path=video_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            output_data = {
                "video": str(video_path),
                "frames_used": result.frames_used,
                "model": result.model_id,
                "caption": result.caption,
            }

            output_path = args.output.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")

            uploaded = None
            if not args.skip_minio_upload:
                if not minio_client:
                    raise ValueError("MinIO client is required for uploading results")
                key = args.result_json_key or (
                    f"{args.video_id}/video_caption.json" if args.video_id else output_path.name
                )
                minio_client.upload_file(str(output_path), args.results_json_bucket, key)
                register_artifact(
                    db,
                    video_id=args.video_id,
                    bucket=args.results_json_bucket,
                    key=key,
                    artifact_type="video_caption.json",
                )
                uploaded = {"bucket": args.results_json_bucket, "key": key}

            update_statuses(
                db,
                job=job,
                video=video,
                job_status=JobStatus.DONE,
                video_status=VideoStatus.DONE,
                progress=100,
            )
            return {
                "caption": result.caption,
                "output_path": str(output_path),
                "uploaded": uploaded,
                "frames_used": result.frames_used,
                "model": result.model_id,
            }
    except Exception as exc:  # pragma: no cover - error path
        update_statuses(
            db,
            job=job,
            video=video,
            job_status=JobStatus.FAILED,
            video_status=VideoStatus.FAILED,
            error=str(exc),
        )
        raise
    finally:
        if db is not None:
            db.close()


# CLI -----------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate detailed captions for a video")
    parser.add_argument("--video", type=Path, default=None, help="Path to a local video")
    parser.add_argument("--video-url", type=str, default=None, help="MinIO HTTP/S3 URL to the video")
    parser.add_argument("--job-id", type=str, default=None, help="Job ID for status updates")
    parser.add_argument("--video-id", type=str, default=None, help="Video ID for status updates")
    parser.add_argument("--database-url", type=str, default=os.getenv("DATABASE_URL"), help="Postgres DATABASE_URL")
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default=os.getenv("MINIO_ENDPOINT_INTERNAL") or os.getenv("MINIO_ENDPOINT_PUBLIC"),
        help="MinIO endpoint",
    )
    parser.add_argument("--minio-access-key", type=str, default=os.getenv("MINIO_ACCESS_KEY"))
    parser.add_argument("--minio-secret-key", type=str, default=os.getenv("MINIO_SECRET_KEY"))
    parser.add_argument(
        "--results-json-bucket",
        type=str,
        default=os.getenv("MINIO_BUCKET_RESULTS_JSON", "results-json"),
        help="Bucket to store caption JSON",
    )
    parser.add_argument("--result-json-key", type=str, default=None, help="Custom key inside results-json bucket")
    parser.add_argument("--skip-minio-upload", action="store_true", help="Do not upload outputs to MinIO")
    parser.add_argument("--output", type=Path, default=Path("caption.json"), help="Where to store the caption JSON")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu/cuda)")
    parser.add_argument("--max-new-tokens", type=int, default=320, help="Maximum tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    return parser


def validate_inputs(args: argparse.Namespace) -> None:
    if args.video_url is None:
        if args.video is None:
            raise ValueError("Provide either --video or --video-url")
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")


async def async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_inputs(args)
    await run_workflow(args)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
