"""CRUD operations for content translations."""
from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.content_translation import ContentTranslation
from src.schemas.content_translation import ContentTranslationCreate, ContentTranslationUpdate


async def get_content_translation(db: AsyncSession, translation_id: UUID) -> Optional[ContentTranslation]:
    """Get a content translation by ID with lesson loaded."""
    stmt = select(ContentTranslation).where(ContentTranslation.translation_id == translation_id).options(
        selectinload(ContentTranslation.lesson)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_content_translation_by_lesson_and_language(
    db: AsyncSession,
    lesson_id: UUID,
    language_code: str
) -> Optional[ContentTranslation]:
    """Get a content translation by lesson ID and language code."""
    stmt = select(ContentTranslation).where(
        ContentTranslation.lesson_id == lesson_id,
        ContentTranslation.language_code == language_code
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_content_translations_by_lesson(
    db: AsyncSession,
    lesson_id: UUID,
    skip: int = 0,
    limit: int = 100
) -> List[ContentTranslation]:
    """Get all translations for a specific lesson."""
    stmt = select(ContentTranslation).where(
        ContentTranslation.lesson_id == lesson_id
    ).options(
        selectinload(ContentTranslation.lesson)
    ).offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def get_content_translations_by_language(
    db: AsyncSession,
    language_code: str,
    skip: int = 0,
    limit: int = 100
) -> List[ContentTranslation]:
    """Get all translations in a specific language."""
    stmt = select(ContentTranslation).where(
        ContentTranslation.language_code == language_code
    ).options(
        selectinload(ContentTranslation.lesson)
    ).offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_content_translation(db: AsyncSession, translation: ContentTranslationCreate) -> ContentTranslation:
    """Create a new content translation."""
    db_translation = ContentTranslation(**translation.model_dump())
    db.add(db_translation)
    await db.commit()
    await db.refresh(db_translation)
    return db_translation


async def update_content_translation(
    db: AsyncSession,
    db_translation: ContentTranslation,
    translation_in: ContentTranslationUpdate
) -> ContentTranslation:
    """Update a content translation."""
    update_data = translation_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_translation, field, value)
    await db.commit()
    await db.refresh(db_translation)
    return db_translation


async def delete_content_translation(db: AsyncSession, translation_id: UUID) -> bool:
    """Delete a content translation."""
    stmt = select(ContentTranslation).where(ContentTranslation.translation_id == translation_id)
    result = await db.execute(stmt)
    db_translation = result.scalar_one_or_none()
    if db_translation:
        await db.delete(db_translation)
        await db.commit()
        return True
    return False