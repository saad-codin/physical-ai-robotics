from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.chatbot_query import ChatbotQuery
from src.schemas.chatbot_query import ChatbotQueryCreate


async def get_chatbot_query(db: AsyncSession, query_id: UUID) -> Optional[ChatbotQuery]:
    """Get a chatbot query by ID with user loaded."""
    stmt = select(ChatbotQuery).where(ChatbotQuery.query_id == query_id).options(
        selectinload(ChatbotQuery.user)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_chatbot_queries(
    db: AsyncSession,
    user_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100
) -> List[ChatbotQuery]:
    """Get a list of chatbot queries, optionally filtered by user."""
    stmt = select(ChatbotQuery).options(
        selectinload(ChatbotQuery.user)
    )

    if user_id:
        stmt = stmt.where(ChatbotQuery.user_id == user_id)

    stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


async def create_chatbot_query(db: AsyncSession, chatbot_query: ChatbotQueryCreate) -> ChatbotQuery:
    """Create a new chatbot query."""
    db_chatbot_query = ChatbotQuery(**chatbot_query.model_dump())
    db.add(db_chatbot_query)
    await db.commit()
    await db.refresh(db_chatbot_query)
    return db_chatbot_query


async def delete_chatbot_query(db: AsyncSession, query_id: UUID) -> bool:
    """Delete a chatbot query."""
    stmt = select(ChatbotQuery).where(ChatbotQuery.query_id == query_id)
    result = await db.execute(stmt)
    db_chatbot_query = result.scalar_one_or_none()
    if db_chatbot_query:
        await db.delete(db_chatbot_query)
        await db.commit()
        return True
    return False