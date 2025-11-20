from fastapi import APIRouter, Depends, HTTPException, Response, Request
from sqlalchemy.orm import Session
from ..db import SessionLocal
from ..models.user import User
from ..security import (
    hash_password,
    verify_password,
    create_session,
    clear_session,
    get_current_user,
)
from ..schemas.auth import RegisterIn, LoginIn


router = APIRouter(prefix="/auth", tags=["auth"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    u = User(email=body.email, password_hash=hash_password(body.password))
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"id": str(u.id), "email": u.email}


@router.post("/login")
def login(
    body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)
):
    u = db.query(User).filter(User.email == body.email).first()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    create_session(db, u, request, response)
    return {"id": str(u.id), "email": u.email}


@router.post("/logout", status_code=204)
def logout(response: Response):
    clear_session(response)
    return Response(status_code=204)


@router.get("/me")
def me(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(db, request)
    if not user:
        raise HTTPException(401, "Unauthorized")
    return {"id": str(user.id), "email": user.email}
