from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.user import UserCreate, UserUpdate, UserResponse as UserRead
from src.schemas.auth import Token, LoginRequest
from src.crud import user as user_crud
from src.core.security import verify_password, get_password_hash
from src.core.auth import create_access_token, get_current_active_user
from src.models.user import User as DBUser  # Alias to avoid conflict with schema UserRead
from src.config import settings

router = APIRouter()

@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    user = await user_crud.get_user_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    hashed_password = get_password_hash(user_in.password)
    user_in_db = user_in.model_dump()
    user_in_db["password_hash"] = hashed_password
    del user_in_db["password"]

    # Handle specialization, ros_experience_level, focus_area during creation
    user_data = {
        **user_in_db,
        "specialization": user_in.specialization or [],
        "ros_experience_level": user_in.ros_experience_level.value if hasattr(user_in.ros_experience_level, "value") else user_in.ros_experience_level,
        "focus_area": user_in.focus_area.value if hasattr(user_in.focus_area, "value") else user_in.focus_area,
        "language_preference": user_in.language_preference,
        "is_active": True  # New users are active by default
    }

    try:
        return await user_crud.create_user(db, user=user_data)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {str(e)}"
        )


@router.post("/login", response_model=Token)
async def login_for_access_token(user_in: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await user_crud.get_user_by_email(db, email=user_in.email)
    if not user or not verify_password(user_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(hours=settings.auth_token_expiry_hours)
    access_token = create_access_token(data={"sub": str(user.user_id)}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserRead)
async def read_users_me(current_user: DBUser = Depends(get_current_active_user)):
    return current_user

@router.put("/me", response_model=UserRead)
async def update_user_me(
    user_in: UserUpdate,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    user_data = user_in.model_dump(exclude_unset=True)
    if user_data.get("password"):
        user_data["password_hash"] = get_password_hash(user_data["password"])
        del user_data["password"]

    return await user_crud.update_user(db, db_user=current_user, user_in=user_data)
