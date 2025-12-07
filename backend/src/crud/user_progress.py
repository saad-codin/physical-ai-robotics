from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.user_progress import UserProgress
from src.schemas.user_progress import UserProgressCreate, UserProgressUpdate


async def get_user_progress(db: AsyncSession, progress_id: UUID) -> Optional[UserProgress]:
    """Get user progress by ID with user and lesson loaded."""
    stmt = select(UserProgress).where(UserProgress.progress_id == progress_id).options(
        selectinload(UserProgress.user),
        selectinload(UserProgress.lesson)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_progress_by_user_and_lesson(
    db: AsyncSession,
    user_id: UUID,
    lesson_id: UUID
) -> Optional[UserProgress]:
    """Get user progress for a specific lesson."""
    stmt = select(UserProgress).where(
        UserProgress.user_id == user_id,
        UserProgress.lesson_id == lesson_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_progresses(
    db: AsyncSession,
    user_id: Optional[UUID] = None,
    lesson_id: Optional[UUID] = None,
    bookmarked: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100
) -> List[UserProgress]:
    """Get a list of user progresses, optionally filtered."""
    stmt = select(UserProgress).options(
        selectinload(UserProgress.user),
        selectinload(UserProgress.lesson)
    )

    if user_id:
        stmt = stmt.where(UserProgress.user_id == user_id)
    if lesson_id:
        stmt = stmt.where(UserProgress.lesson_id == lesson_id)
    if bookmarked is not None:
        stmt = stmt.where(UserProgress.bookmarked == bookmarked)

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_user_progress(db: AsyncSession, user_progress: UserProgressCreate) -> UserProgress:
    """Create a new user progress record."""
    db_user_progress = UserProgress(**user_progress.model_dump())
    db.add(db_user_progress)
    await db.commit()
    await db.refresh(db_user_progress)
    return db_user_progress


async def update_user_progress(
    db: AsyncSession,
    db_user_progress: UserProgress,
    user_progress_in: UserProgressUpdate
) -> UserProgress:
    """Update a user progress record."""
    update_data = user_progress_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user_progress, field, value)
    await db.commit()
    await db.refresh(db_user_progress)
    return db_user_progress


async def delete_user_progress(db: AsyncSession, progress_id: UUID) -> bool:
    """Delete a user progress record."""
    stmt = select(UserProgress).where(UserProgress.progress_id == progress_id)
    result = await db.execute(stmt)
    db_user_progress = result.scalar_one_or_none()
    if db_user_progress:
        await db.delete(db_user_progress)
        await db.commit()
        return True
    return False