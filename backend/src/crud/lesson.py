from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.lesson import Lesson
from src.schemas.lesson import LessonCreate, LessonUpdate


async def get_lesson(db: AsyncSession, lesson_id: UUID) -> Optional[Lesson]:
    """Get a lesson by ID with module, progress, embeddings, and translations loaded."""
    stmt = select(Lesson).where(Lesson.lesson_id == lesson_id).options(
        selectinload(Lesson.module),
        selectinload(Lesson.progress),
        selectinload(Lesson.embeddings),
        selectinload(Lesson.translations)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_lessons(
    db: AsyncSession,
    module_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Lesson]:
    """Get a list of lessons, optionally filtered by module ID."""
    stmt = select(Lesson).options(
        selectinload(Lesson.module)
    ).order_by(Lesson.order_index)

    if module_id:
        stmt = stmt.where(Lesson.module_id == module_id)

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_lesson(db: AsyncSession, lesson: LessonCreate) -> Lesson:
    """Create a new lesson."""
    db_lesson = Lesson(**lesson.model_dump())
    db.add(db_lesson)
    await db.commit()
    await db.refresh(db_lesson)
    return db_lesson


async def update_lesson(db: AsyncSession, db_lesson: Lesson, lesson_in: LessonUpdate) -> Lesson:
    """Update a lesson."""
    update_data = lesson_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_lesson, field, value)
    await db.commit()
    await db.refresh(db_lesson)
    return db_lesson


async def delete_lesson(db: AsyncSession, lesson_id: UUID) -> bool:
    """Delete a lesson."""
    stmt = select(Lesson).where(Lesson.lesson_id == lesson_id)
    result = await db.execute(stmt)
    db_lesson = result.scalar_one_or_none()
    if db_lesson:
        await db.delete(db_lesson)
        await db.commit()
        return True
    return False