"""Chatbot query schema definitions for Pydantic models."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime


class ChatbotQueryBase(BaseModel):
    """Base chatbot query schema."""
    user_id: Optional[UUID] = None
    query_text: str
    retrieved_passages: List[Dict[str, Any]] = []
    response_text: str
    response_generation_time_ms: int


class ChatbotQueryCreate(ChatbotQueryBase):
    """Schema for creating a new chatbot query."""
    query_text: str
    response_generation_time_ms: int
    # user_id will be set from the authenticated user


class ChatbotQueryUpdate(BaseModel):
    """Schema for updating a chatbot query."""
    query_text: Optional[str] = None
    response_text: Optional[str] = None


class ChatbotQueryRead(ChatbotQueryBase):
    """Schema for chatbot query response."""
    query_id: UUID
    user_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True