"""Lesson and module schemas for request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID
from datetime import datetime
from enum import Enum


class ModuleNameEnum(str, Enum):
    """Module name enum for validation."""
    ROS2 = "ROS2"
    DIGITAL_TWIN = "DigitalTwin"
    AI_ROBOT_BRAIN = "AIRobotBrain"
    VLA = "VLA"


class CodeExample(BaseModel):
    """Schema for code example embedded in lesson."""
    language: str
    code: str
    description: str


class ModuleBase(BaseModel):
    """Base module schema."""
    name: ModuleNameEnum
    description: str
    order_index: int


class ModuleCreate(ModuleBase):
    """Schema for creating a module."""
    pass


class ModuleResponse(ModuleBase):
    """Schema for module response."""
    module_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LessonBase(BaseModel):
    """Base lesson schema."""
    module_id: UUID
    title: str = Field(..., max_length=500)
    learning_objectives: list[str] = Field(default_factory=list)
    content_markdown: str
    code_examples: list[CodeExample] = Field(default_factory=list)
    diagrams: list[str] = Field(default_factory=list)
    order_index: int


class LessonCreate(LessonBase):
    """Schema for creating a lesson."""
    pass


class LessonUpdate(BaseModel):
    """Schema for updating a lesson."""
    title: Optional[str] = None
    learning_objectives: Optional[list[str]] = None
    content_markdown: Optional[str] = None
    code_examples: Optional[list[CodeExample]] = None
    diagrams: Optional[list[str]] = None
    order_index: Optional[int] = None


class LessonResponse(LessonBase):
    """Schema for lesson response."""
    lesson_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LessonDetailResponse(LessonResponse):
    """Schema for detailed lesson response with module information."""
    module: ModuleResponse

    class Config:
        from_attributes = True


# Alias for backward compatibility
LessonRead = LessonResponse
