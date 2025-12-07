"""Module schema definitions for Pydantic models."""
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from enum import Enum
from datetime import datetime


class ModuleName(str, Enum):
    """Enumeration of available module names."""
    ROS_FUNDAMENTALS = "ROS Fundamentals"
    DIGITAL_TWIN_SIMULATION = "Digital Twin & Simulation"
    AI_ROBOT_BRAIN = "AI-Robot Brain"
    VISION_LANGUAGE_ACTION = "Vision Language Action"


class ModuleBase(BaseModel):
    """Base module schema."""
    name: ModuleName
    title: str
    description: str
    order_index: int = 0
    is_active: bool = True


class ModuleCreate(ModuleBase):
    """Schema for creating a new module."""
    pass


class ModuleUpdate(BaseModel):
    """Schema for updating an existing module."""
    name: Optional[ModuleName] = None
    title: Optional[str] = None
    description: Optional[str] = None
    order_index: Optional[int] = None
    is_active: Optional[bool] = None


class ModuleResponse(ModuleBase):
    """Schema for module response."""
    module_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Alias for backward compatibility
ModuleRead = ModuleResponse