from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..models.video import Video
from ..security import get_current_user
from ..schemas.video import VideoOut


router = APIRouter(prefix="/videos", tags=["videos"])


def get_db():
    db = SessionLocal();
    try: yield db
    finally: db.close()


@router.get("/me")
def my_videos(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    if not user:
        raise HTTPException(401, "Unauthorized")
    items = db.query(Video).filter(Video.user_id==user.id).order_by(Video.created_at.desc()).all()
    return {"items": [VideoOut.model_validate(i).model_dump() for i in items]}


@router.get("/{video_id}")
def get_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    if not user:
        raise HTTPException(401, "Unauthorized")
    v = db.get(Video, video_id)
    if not v or v.user_id != user.id:
        raise HTTPException(404, "Not found")
    return VideoOut.model_validate(v)