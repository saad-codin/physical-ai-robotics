from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.user import User
from src.schemas.user import UserCreate, UserUpdate


async def get_user(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """Get a user by ID with progress and chatbot queries loaded."""
    stmt = select(User).where(User.user_id == user_id).options(
        selectinload(User.progress),
        selectinload(User.chatbot_queries)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get a user by email."""
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100):
    """Get a list of users."""
    stmt = select(User).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_user(db: AsyncSession, user: dict) -> User:
    """Create a new user."""
    db_user = User(**user)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def update_user(db: AsyncSession, db_user: User, user_in: dict) -> User:
    """Update a user."""
    for field, value in user_in.items():
        setattr(db_user, field, value)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def delete_user(db: AsyncSession, user_id: UUID) -> bool:
    """Delete a user."""
    stmt = select(User).where(User.user_id == user_id)
    result = await db.execute(stmt)
    db_user = result.scalar_one_or_none()
    if db_user:
        await db.delete(db_user)
        await db.commit()
        return True
    return False