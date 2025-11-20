import asyncio, json
from fastapi import Response


async def sse_event_stream(pubsub):
    try:
        while True:
            message = await pubsub.get() # абстракция; подключите Redis pub/sub
            yield f"event: {message['event']}\n" + f"data: {json.dumps(message['data'])}\n\n"
    except asyncio.CancelledError:
        return


def sse_headers() -> dict:
    return {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    }

def to_sse_frame(json_str: str) -> str:
    # json_str имеет {"event": "...","data": {...}}
    return f"data: {json_str}\n\n"  # event можно инлайнить в data