from pydantic import BaseModel


class JobOut(BaseModel):
    id: str
    status: str
    progress: int
    error: str | None = None
