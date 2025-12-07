"""Personalization schemas for user progress and recommendations."""
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


class LessonRecommendation(BaseModel):
    """Schema for a recommended lesson."""
    lesson_id: UUID
    title: str
    module_name: str
    personalization_score: float = Field(..., ge=0.0, le=1.0)
    relevance_reasons: list[str]


class LessonRecommendationsResponse(BaseModel):
    """Schema for lesson recommendations response."""
    recommendations: list[LessonRecommendation]


class UserProgressSummary(BaseModel):
    """Schema for user progress summary."""
    total_lessons: int
    completed_lessons: int
    bookmarked_lessons: int
    total_time_spent_seconds: int
    completion_percentage: float = Field(..., ge=0.0, le=100.0)
    modules_progress: dict[str, dict[str, int]]  # {module_name: {total, completed}}


class PersonalizedLearningPathResponse(BaseModel):
    """Schema for personalized learning path response."""
    modules: list[dict]
    recommended_next: Optional[dict] = None


class UserProgressSummaryResponse(BaseModel):
    """Schema for user progress summary response."""
    user_id: UUID
    summary: UserProgressSummary
