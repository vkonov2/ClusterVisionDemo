# Minimal image for running llava_video_gui.py
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY llava_video_gui.py requirements-llava.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-llava.txt

EXPOSE 7860
CMD ["python", "llava_video_gui.py", "--server-port", "7860"]
