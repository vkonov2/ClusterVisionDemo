from fastapi import FastAPI
from .routers import health, auth, uploads, videos, jobs, events


app = FastAPI(title="SynapSight API")
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(uploads.router)
app.include_router(videos.router)
app.include_router(jobs.router)
app.include_router(events.router)