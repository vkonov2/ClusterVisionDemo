import os, json
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from app.models.artifact import Artifact
import boto3
from botocore.client import Config


INTERNAL = os.getenv("MINIO_ENDPOINT_INTERNAL")
AK = os.getenv("MINIO_ACCESS_KEY")
SK = os.getenv("MINIO_SECRET_KEY")
BUCKET_RESULTS = os.getenv("MINIO_BUCKET_RESULTS")


s3 = boto3.client("s3", endpoint_url=INTERNAL, aws_access_key_id=AK, aws_secret_access_key=SK, config=Config(signature_version="s3v4"))


def save_artifact(db: Session, video_id: str, typ: str, key: str, data: dict):
    body = json.dumps(data).encode()
    s3.put_object(Bucket=BUCKET_RESULTS, Key=key, Body=body, ContentType="application/json")
    a = Artifact(video_id=video_id, type=typ, uri=f"s3://{BUCKET_RESULTS}/{key}")
    db.add(a); db.commit()