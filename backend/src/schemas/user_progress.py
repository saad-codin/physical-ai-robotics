"""User progress schema definitions for Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime


class UserProgressBase(BaseModel):
    """Base user progress schema."""
    lesson_id: UUID
    completed_at: Optional[datetime] = None
    bookmarked: bool = False
    time_spent_seconds: int = Field(default=0, ge=0)


class UserProgressCreate(BaseModel):
    """Schema for creating user progress entry."""
    lesson_id: UUID


class UserProgressUpdate(BaseModel):
    """Schema for updating user progress."""
    completed_at: Optional[datetime] = None
    bookmarked: Optional[bool] = None
    time_spent_seconds: Optional[int] = Field(None, ge=0)


class UserProgressResponse(UserProgressBase):
    """Schema for user progress response."""
    progress_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Alias for backward compatibility
UserProgressRead = UserProgressResponse