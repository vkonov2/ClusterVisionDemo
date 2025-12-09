# Call examples for the video2text container

## cURL (MinIO URL)
```bash
curl -X POST http://localhost:8000/caption \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "http://minio:9000/raw-videos/demo.mp4",
    "job_id": "123",
    "video_id": "456",
    "database_url": "postgresql+psycopg2://user:pass@postgres:5432/db",
    "minio_endpoint": "http://minio:9000",
    "minio_access_key": "minio",
    "minio_secret_key": "minio123",
    "results_json_bucket": "results-json"
  }'
```

## Python client
```python
import requests

payload = {
    "video_url": "http://minio:9000/raw-videos/demo.mp4",
    "job_id": "123",
    "video_id": "456",
}
response = requests.post("http://localhost:8000/caption", json=payload, timeout=600)
response.raise_for_status()
print(response.json())
```

The service responds with the generated caption, the local path of the saved JSON file, upload info (if enabled), and metadata on how many frames were processed.
