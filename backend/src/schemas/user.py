"""User schemas for request/response validation."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import Enum


class ROSExperienceLevelEnum(str, Enum):
    """ROS experience level enum for validation."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class FocusAreaEnum(str, Enum):
    """Focus area enum for validation."""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    BOTH = "both"


class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr
    specialization: list[str] = Field(default_factory=list)
    ros_experience_level: ROSExperienceLevelEnum = ROSExperienceLevelEnum.BEGINNER
    focus_area: FocusAreaEnum = FocusAreaEnum.BOTH
    language_preference: str = "en"


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=128)


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    specialization: Optional[list[str]] = None
    ros_experience_level: Optional[ROSExperienceLevelEnum] = None
    focus_area: Optional[FocusAreaEnum] = None
    language_preference: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response."""
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
