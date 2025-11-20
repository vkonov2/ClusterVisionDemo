import os
import boto3
from botocore.client import Config


INTERNAL = os.getenv("MINIO_ENDPOINT_INTERNAL")
PUBLIC = os.getenv("MINIO_ENDPOINT_PUBLIC")
REGION = os.getenv("MINIO_REGION", "us-east-1")
AK = os.getenv("MINIO_ACCESS_KEY")
SK = os.getenv("MINIO_SECRET_KEY")
BUCKET_UPLOADS = os.getenv("MINIO_BUCKET_UPLOADS")
BUCKET_RESULTS = os.getenv("MINIO_BUCKET_RESULTS")


# для операций API/бекенда
s3_internal = boto3.client(
    "s3",
    endpoint_url=INTERNAL,
    aws_access_key_id=AK,
    aws_secret_access_key=SK,
    region_name=REGION,
    config=Config(signature_version="s3v4"),
)


# отдельный клиент для ПОДПИСАННЫХ URL (должен указывать на публичный хост)
s3_public = boto3.client(
    "s3",
    endpoint_url=PUBLIC,
    aws_access_key_id=AK,
    aws_secret_access_key=SK,
    region_name=REGION,
    config=Config(signature_version="s3v4"),
)
