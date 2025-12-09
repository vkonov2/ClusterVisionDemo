"""FastAPI entrypoint for the video2text captioning service."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import analytics

app = FastAPI(title="video2text", description="UGC video captioning service")


class CaptionRequest(BaseModel):
    video_url: Optional[str] = Field(
        default=None, description="MinIO HTTP/S3 URL pointing to the source video"
    )
    video_path: Optional[str] = Field(default=None, description="Local path inside the container")
    job_id: Optional[str] = None
    video_id: Optional[str] = None
    database_url: Optional[str] = Field(default=os.getenv("DATABASE_URL"))
    minio_endpoint: Optional[str] = Field(
        default=os.getenv("MINIO_ENDPOINT_INTERNAL") or os.getenv("MINIO_ENDPOINT_PUBLIC")
    )
    minio_access_key: Optional[str] = Field(default=os.getenv("MINIO_ACCESS_KEY"))
    minio_secret_key: Optional[str] = Field(default=os.getenv("MINIO_SECRET_KEY"))
    results_json_bucket: str = Field(
        default=os.getenv("MINIO_BUCKET_RESULTS_JSON", "results-json"),
        description="Bucket to store caption JSON",
    )
    result_json_key: Optional[str] = None
    skip_minio_upload: bool = False
    device: Optional[str] = None
    max_new_tokens: int = Field(default=320, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class CaptionResponse(BaseModel):
    caption: str
    output_path: str
    uploaded: Optional[dict[str, str]] = None
    frames_used: int
    model: str


@app.post("/caption", response_model=CaptionResponse)
async def caption_video(payload: CaptionRequest) -> Any:
    if payload.video_url is None and payload.video_path is None:
        raise HTTPException(status_code=400, detail="Provide video_url or video_path")

    args = analytics.build_arg_parser().parse_args([])
    args.video_url = payload.video_url
    args.video = Path(payload.video_path) if payload.video_path else None
    args.job_id = payload.job_id
    args.video_id = payload.video_id
    args.database_url = payload.database_url
    args.minio_endpoint = payload.minio_endpoint
    args.minio_access_key = payload.minio_access_key
    args.minio_secret_key = payload.minio_secret_key
    args.results_json_bucket = payload.results_json_bucket
    args.result_json_key = payload.result_json_key
    args.skip_minio_upload = payload.skip_minio_upload
    args.output = Path("caption.json")
    args.device = payload.device
    args.max_new_tokens = payload.max_new_tokens
    args.temperature = payload.temperature
    args.top_p = payload.top_p

    try:
        analytics.validate_inputs(args)
    except Exception as exc:  # pragma: no cover - FastAPI validation path
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await analytics.run_workflow(args)
    except Exception as exc:  # pragma: no cover - runtime error path
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CaptionResponse(**result)


@app.get("/healthz")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
