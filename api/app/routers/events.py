from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from ..security import get_current_user
from ..db import SessionLocal
from ..sse import sse_headers, to_sse_frame
from ..pubsub import subscribe
import asyncio

router = APIRouter(prefix="/events", tags=["events"])

@router.get("")
async def events(request: Request):
    db = SessionLocal()
    try:
        user = get_current_user(db, request)
        if not user:
            raise HTTPException(401, "Unauthorized")

        async def stream():
            # keepalive раз в N секунд + реальные сообщения из pub/sub
            keepalive = asyncio.create_task(_keepalive())
            try:
                async for msg in subscribe(str(user.id)):
                    # Клиент отключился?
                    if await request.is_disconnected():
                        break
                    yield to_sse_frame(msg)
            finally:
                keepalive.cancel()

        return StreamingResponse(stream(), headers=sse_headers())
    finally:
        db.close()

async def _keepalive():
    while True:
        yield "event: keepalive\ndata: {}\n\n"
        await asyncio.sleep(15)
