"""LessonEmbedding schemas for vector storage and RAG operations."""
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List, Optional


class LessonEmbeddingBase(BaseModel):
    """Base lesson embedding schema."""
    lesson_id: UUID
    passage_text: str = Field(..., min_length=1, max_length=8192)  # Up to 8KB per passage
    qdrant_vector_id: str = Field(..., min_length=1, max_length=255)  # Reference to Qdrant vector ID


class LessonEmbeddingCreate(LessonEmbeddingBase):
    """Schema for creating a lesson embedding."""
    pass


class LessonEmbeddingUpdate(BaseModel):
    """Schema for updating a lesson embedding."""
    passage_text: Optional[str] = Field(None, min_length=1, max_length=8192)
    qdrant_vector_id: Optional[str] = Field(None, min_length=1, max_length=255)


class LessonEmbeddingRead(LessonEmbeddingBase):
    """Schema for lesson embedding response."""
    embedding_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LessonEmbeddingSearchRequest(BaseModel):
    """Schema for embedding search request."""
    query_text: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)  # Number of results to return
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)  # Minimum similarity score


class LessonEmbeddingSearchResponse(BaseModel):
    """Schema for embedding search response."""
    results: List[dict]  # Contains lesson_id, passage_text, similarity_score
    query_text: str
    execution_time_ms: int