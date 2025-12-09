# video2text service

This Docker service wraps the `openinterx/UGC-VideoCaptioner` model to produce **text-only** captions for UGC-style videos. It mirrors the existing analytics containers by working with MinIO + Postgres, but it disables audio synthesis to keep inference fast.

## Folder layout
```
video2text/
├── Dockerfile
├── README.md
├── call_api_example.md
└── app
    ├── analytics.py      # CLI entrypoint for batch jobs
    ├── api.py            # FastAPI service
    ├── models/           # Place pre-downloaded model weights here (optional)
    └── requirements.txt
```

## Environment
The service loads the same env chain as other tools:
- `.env` and `.env.local` from the repo root
- `video2text/app/.env.local-video2text` (optional)

Expected variables when running inside Docker:
- `DATABASE_URL` – Postgres connection for job/video status updates
- `MINIO_ENDPOINT_INTERNAL`/`MINIO_ENDPOINT_PUBLIC`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `MINIO_BUCKET_RESULTS_JSON` – bucket for storing caption JSON (default `results-json`)

## Pre-downloading the model (optional)
To avoid first-run downloads in production, fetch the model into the `app/models` directory and mount it to the Hugging Face cache inside the container:
```bash
mkdir -p video2text/app/models
huggingface-cli download openinterx/UGC-VideoCaptioner --local-dir video2text/app/models/UGC-VideoCaptioner
```
Then run the container with
```bash
-v $(pwd)/video2text/app/models/UGC-VideoCaptioner:/root/.cache/huggingface/hub/models--openinterx--UGC-VideoCaptioner
```

## Local CLI usage
```bash
python video2text/app/analytics.py \
  --video /data/demo.mp4 \
  --database-url $DATABASE_URL \
  --minio-endpoint $MINIO_ENDPOINT_INTERNAL \
  --minio-access-key $MINIO_ACCESS_KEY \
  --minio-secret-key $MINIO_SECRET_KEY
```

The script downloads the video from MinIO when `--video-url` is provided, writes `caption.json`, optionally uploads it to the `results-json` bucket, and updates job/video rows.

## Running the Docker image
```bash
docker build -t video2text:latest video2text
# Run with access to MinIO/Postgres environment
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=$DATABASE_URL \
  -e MINIO_ENDPOINT_INTERNAL=$MINIO_ENDPOINT_INTERNAL \
  -e MINIO_ACCESS_KEY=$MINIO_ACCESS_KEY \
  -e MINIO_SECRET_KEY=$MINIO_SECRET_KEY \
  video2text:latest
```

The FastAPI server exposes `POST /caption` and `GET /healthz`.
