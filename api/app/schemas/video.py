from pydantic import BaseModel


class VideoOut(BaseModel):
    id: str
    filename: str
    status: str
    size_bytes: int | None = None

    class Config:
        from_attributes = True
