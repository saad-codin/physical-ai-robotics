from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.lesson_embedding import LessonEmbedding
from src.schemas.lesson_embedding import LessonEmbeddingCreate, LessonEmbeddingUpdate


async def get_lesson_embedding(db: AsyncSession, embedding_id: UUID) -> Optional[LessonEmbedding]:
    """Get a lesson embedding by ID with lesson loaded."""
    stmt = select(LessonEmbedding).where(LessonEmbedding.embedding_id == embedding_id).options(
        selectinload(LessonEmbedding.lesson)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_lesson_embeddings_by_lesson(
    db: AsyncSession,
    lesson_id: UUID,
    skip: int = 0,
    limit: int = 100
) -> List[LessonEmbedding]:
    """Get embeddings for a specific lesson."""
    stmt = select(LessonEmbedding).where(
        LessonEmbedding.lesson_id == lesson_id
    ).options(
        selectinload(LessonEmbedding.lesson)
    ).offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def get_lesson_embedding_by_qdrant_vector_id(
    db: AsyncSession,
    qdrant_vector_id: str
) -> Optional[LessonEmbedding]:
    """Get an embedding by Qdrant vector ID."""
    stmt = select(LessonEmbedding).where(
        LessonEmbedding.qdrant_vector_id == qdrant_vector_id
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def create_lesson_embedding(db: AsyncSession, lesson_embedding: LessonEmbeddingCreate) -> LessonEmbedding:
    """Create a new lesson embedding."""
    db_lesson_embedding = LessonEmbedding(**lesson_embedding.model_dump())
    db.add(db_lesson_embedding)
    await db.commit()
    await db.refresh(db_lesson_embedding)
    return db_lesson_embedding


async def update_lesson_embedding(
    db: AsyncSession,
    db_lesson_embedding: LessonEmbedding,
    lesson_embedding_in: LessonEmbeddingUpdate
) -> LessonEmbedding:
    """Update a lesson embedding."""
    update_data = lesson_embedding_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_lesson_embedding, field, value)
    await db.commit()
    await db.refresh(db_lesson_embedding)
    return db_lesson_embedding


async def delete_lesson_embedding(db: AsyncSession, embedding_id: UUID) -> bool:
    """Delete a lesson embedding."""
    stmt = select(LessonEmbedding).where(LessonEmbedding.embedding_id == embedding_id)
    result = await db.execute(stmt)
    db_lesson_embedding = result.scalar_one_or_none()
    if db_lesson_embedding:
        await db.delete(db_lesson_embedding)
        await db.commit()
        return True
    return False


async def delete_lesson_embeddings_by_lesson(db: AsyncSession, lesson_id: UUID) -> int:
    """Delete all embeddings for a specific lesson."""
    stmt = select(LessonEmbedding).where(LessonEmbedding.lesson_id == lesson_id)
    result = await db.execute(stmt)
    embeddings = result.scalars().all()

    for embedding in embeddings:
        await db.delete(embedding)

    await db.commit()
    return len(embeddings)