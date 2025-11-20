from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from ..db import Base


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    type = Column(String, nullable=False)  # 'poster','ffprobe.json','audio.json', ...
    uri = Column(String, nullable=False)  # s3://bucket/key или https://...
    meta_json = Column(String)  # строка JSON (минимум для MVP)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
