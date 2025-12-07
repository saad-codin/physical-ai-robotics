"""Module model for textbook content organization."""
from sqlalchemy import Column, String, Integer, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import enum
from src.db.base import Base, TimestampMixin


class ModuleName(str, enum.Enum):
    """Module name enum matching the four-module structure."""
    ROS2 = "ROS2"
    DIGITAL_TWIN = "DigitalTwin"
    AI_ROBOT_BRAIN = "AIRobotBrain"
    VLA = "VLA"


class Module(Base, TimestampMixin):
    """Module entity representing one of the four textbook modules.

    Constitution Principle IV: Content Structure - Four-Module Format
    - ROS 2 Fundamentals
    - Digital Twin & Simulation
    - AI-Robot Brain
    - Vision Language Action (VLA)
    """
    __tablename__ = "modules"

    module_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(SQLEnum(ModuleName), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)
    order_index = Column(Integer, nullable=False, unique=True)

    # Relationships
    lessons = relationship("Lesson", back_populates="module", cascade="all, delete-orphan", order_by="Lesson.order_index")

    def __repr__(self) -> str:
        return f"<Module(module_id={self.module_id}, name={self.name})>"
