import os, json, asyncio
import redis.asyncio as redis

REDIS_URL_PUBSUB = os.getenv("REDIS_URL_PUBSUB", "redis://redis:6379/3")
r = redis.from_url(REDIS_URL_PUBSUB, decode_responses=True)

def channel_for_user(user_id: str) -> str:
    return f"user:{user_id}:events"

async def publish(user_id: str, event: str, data: dict):
    msg = json.dumps({"event": event, "data": data})
    await r.publish(channel_for_user(user_id), msg)

async def subscribe(user_id: str):
    pubsub = r.pubsub()
    await pubsub.subscribe(channel_for_user(user_id))
    try:
        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue
            yield raw["data"]  # уже JSON-строка
    finally:
        await pubsub.close()
