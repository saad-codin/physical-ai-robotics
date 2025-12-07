"""User model for authentication and profiling."""
from sqlalchemy import Column, String, Enum as SQLEnum, ARRAY, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
import enum
from src.db.base import Base, TimestampMixin


class ROSExperienceLevel(str, enum.Enum):
    """ROS experience level enum."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class FocusArea(str, enum.Enum):
    """User focus area enum."""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    BOTH = "both"


class User(Base, TimestampMixin):
    """User entity for authentication and profiling.

    Stores user credentials, specialization, and preferences for personalization.
    """
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # Optional for OAuth users

    # Specialization fields for personalization
    specialization = Column(ARRAY(String), default=list, nullable=False)
    ros_experience_level = Column(
        SQLEnum(ROSExperienceLevel),
        default=ROSExperienceLevel.BEGINNER,
        nullable=False
    )
    focus_area = Column(
        SQLEnum(FocusArea),
        default=FocusArea.BOTH,
        nullable=False
    )

    # Language preference for multi-language support
    language_preference = Column(String(10), default="en", nullable=False)

    # Account status
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    progress = relationship("UserProgress", back_populates="user", cascade="all, delete-orphan")
    chatbot_queries = relationship("ChatbotQuery", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, email={self.email})>"
