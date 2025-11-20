import os, json, subprocess, uuid
from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.tasks.utils import save_artifact


DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")
BUCKET_UPLOADS = os.getenv("MINIO_BUCKET_UPLOADS")
BUCKET_RESULTS = os.getenv("MINIO_BUCKET_RESULTS")
MINIO_INTERNAL = os.getenv("MINIO_ENDPOINT_INTERNAL")
AK = os.getenv("MINIO_ACCESS_KEY")
SK = os.getenv("MINIO_SECRET_KEY")


@shared_task(name="tasks.analyze")
def analyze(video_id: str, job_id: str):
    from sqlalchemy.orm import Session
    from app.models.job import Job, JobStatus
    from app.models.video import Video, VideoStatus


    db: Session = SessionLocal()
    try:
        job = db.get(Job, job_id)
        vid = db.get(Video, video_id)
        if not job or not vid:
            return
        job.status = JobStatus.STARTED
        vid.status = VideoStatus.PROCESSING
        db.commit()


        # Считать ffprobe через stdin/stdout
        # Вместо скачивания файла: ffprobe умеет HTTP; укажите presigned GET, но для простоты — дадим доступ через s3 API
        # (Для MVP проще: загрузить объект локально и проанализировать; можно оптимизировать позже.)


        # Скачаем из MinIO локально
        import boto3
        from botocore.client import Config
        s3 = boto3.client("s3", endpoint_url=MINIO_INTERNAL, aws_access_key_id=AK, aws_secret_access_key=SK, config=Config(signature_version="s3v4"))
        local_path = f"/tmp/{uuid.uuid4()}-{vid.filename}"
        s3.download_file(BUCKET_UPLOADS, vid.storage_key, local_path)


        cmd = [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", local_path]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)


        # сохранить артефакт в results как JSON
        key = f"{video_id}/ffprobe.json"
        save_artifact(db, video_id, "ffprobe.json", key, info)


        job.status = JobStatus.DONE
        job.progress = 100
        vid.status = VideoStatus.DONE
        db.commit()
    except Exception as e:
        job = db.get(Job, job_id)
        if job:
            from app.models.job import JobStatus
            job.status = JobStatus.FAILED
            job.error = str(e)
            db.commit()
    finally:
        db.close()