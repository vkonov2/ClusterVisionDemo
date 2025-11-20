import os, datetime as dt
from argon2 import PasswordHasher
from itsdangerous import URLSafeSerializer
from fastapi import Response, Request
from sqlalchemy.orm import Session
from .models.user import User
from .models.session import Session as SessionModel


ph = PasswordHasher()
SECRET_KEY = os.getenv("SECRET_KEY")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "sid")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "2592000"))
ser = URLSafeSerializer(SECRET_KEY, salt="session")


# Пароли
hash_password = lambda pwd: ph.hash(pwd)


def verify_password(pwd: str, stored_hash: str) -> bool:
    try:
        return ph.verify(stored_hash, pwd)
    except Exception:
        return False


# Сессии


def create_session(db: Session, user: User, request: Request, response: Response):
    expires_at = dt.datetime.utcnow() + dt.timedelta(seconds=SESSION_TTL_SECONDS)
    s = SessionModel(
        user_id=user.id,
        expires_at=expires_at,
        ip=request.client.host,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    token = ser.dumps({"sid": str(s.id)})
    response.set_cookie(SESSION_COOKIE_NAME, token, httponly=True, samesite="lax")
    return s


def get_current_user(db: Session, request: Request) -> User | None:
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return None
    data = ser.loads(raw)
    sid = data.get("sid")
    if not sid:
        return None
    s = db.get(SessionModel, sid)
    if not s:
        return None
    if s.expires_at < dt.datetime.utcnow():
        return None
    return db.get(User, s.user_id)


def clear_session(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME)
