import os, uuid, datetime as dt
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models.job import Job, JobStatus
from .models.video import Video, VideoStatus


# минимальная имитация публикации; реальная отправка произойдет из worker
from celery import Celery
cel = Celery(__name__, broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND"))


def enqueue_basic_job(video_id: str) -> str:
    db: Session = SessionLocal()
    try:
        j = Job(video_id=video_id, status=JobStatus.PENDING)
        db.add(j); db.commit(); db.refresh(j)
        # Публикация в Celery
        cel.send_task("tasks.analyze", kwargs={"video_id": video_id, "job_id": str(j.id)})
        return str(j.id)
    finally:
        db.close()