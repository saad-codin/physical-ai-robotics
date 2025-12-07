"""Chatbot schemas for RAG query/response validation."""
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class RetrievedPassage(BaseModel):
    """Schema for a retrieved passage from Qdrant."""
    lesson_id: UUID
    passage_text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class ChatbotQueryRequest(BaseModel):
    """Schema for chatbot query request."""
    query_text: str = Field(..., min_length=1, max_length=2000)


class ChatbotQueryResponse(BaseModel):
    """Schema for chatbot query response."""
    query_id: UUID
    query_text: str
    response_text: str
    retrieved_passages: list[RetrievedPassage]
    response_generation_time_ms: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatbotQueryHistory(BaseModel):
    """Schema for chatbot query history item."""
    query_id: UUID
    query_text: str
    response_text: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatbotQueryHistoryResponse(BaseModel):
    """Schema for paginated chatbot query history."""
    queries: list[ChatbotQueryHistory]
    total: int
    page: int
    page_size: int
