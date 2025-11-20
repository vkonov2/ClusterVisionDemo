from pydantic import BaseModel


class InitUploadIn(BaseModel):
    filename: str
    size: int
    mime: str | None = None


class InitUploadOut(BaseModel):
    key: str
    url: str
    fields: dict
