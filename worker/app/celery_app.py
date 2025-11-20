import os
from celery import Celery


celery = Celery(
"synapsight",
broker=os.getenv("CELERY_BROKER_URL"),
backend=os.getenv("CELERY_RESULT_BACKEND"),
)
celery.autodiscover_tasks(["app.tasks"])