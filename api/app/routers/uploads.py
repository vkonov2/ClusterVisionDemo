import uuid, os
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..s3 import s3_public, s3_internal, BUCKET_UPLOADS
from ..models.video import Video, VideoStatus
from ..models.user import User
from ..security import get_current_user
from ..schemas.upload import InitUploadIn, InitUploadOut


router = APIRouter(prefix="/uploads", tags=["uploads"])


MAX_SIZE = 5 * 1024**3


def get_db():
    db = SessionLocal();
    try: yield db
    finally: db.close()


@router.post("/init", response_model=InitUploadOut)
def init_upload(body: InitUploadIn, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    if not user:
        raise HTTPException(401, "Unauthorized")
    if body.size <= 0 or body.size > MAX_SIZE:
        raise HTTPException(400, "Invalid size")
    key = f"{uuid.uuid4()}-{body.filename}"
    presigned = s3_public.generate_presigned_post(
        Bucket=BUCKET_UPLOADS,
        Key=key,
        ExpiresIn=600,
        Conditions=[["content-length-range", 0, min(body.size, MAX_SIZE)]],
    )
    # черновик записи видео
    v = Video(user_id=user.id, filename=body.filename, storage_key=key, status=VideoStatus.UPLOADING, mime=body.mime, size_bytes=body.size)
    db.add(v); db.commit()
    return {"key": key, "url": presigned["url"], "fields": presigned["fields"]}


@router.post("/complete")
def complete_upload(key: str, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    if not user:
        raise HTTPException(401, "Unauthorized")
    v = db.query(Video).filter(Video.user_id==user.id, Video.storage_key==key).first()
    if not v:
        raise HTTPException(404, "Video not found")
    # проверка наличия объекта
    try:
        head = s3_internal.head_object(Bucket=BUCKET_UPLOADS, Key=key)
    except Exception:
        raise HTTPException(400, "Object not found in storage")
    v.size_bytes = head.get("ContentLength")
    v.status = VideoStatus.QUEUED
    db.commit()
    # создать job (минимально) и отправить в Celery
    from ..deps import enqueue_basic_job
    job_id = enqueue_basic_job(video_id=str(v.id))
    return {"video_id": str(v.id), "job_id": job_id}