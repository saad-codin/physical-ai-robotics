"""UserProgress model for tracking lesson completion and bookmarks."""
from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base, TimestampMixin


class UserProgress(Base, TimestampMixin):
    """UserProgress entity tracking user advancement through lessons.

    Supports:
    - Lesson completion tracking
    - Bookmarking lessons
    - Time spent analytics
    - Module roadmap visual indicators
    """
    __tablename__ = "user_progress"

    progress_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("lessons.lesson_id"), nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)  # Null if not completed
    bookmarked = Column(Boolean, default=False, nullable=False)
    time_spent_seconds = Column(Integer, default=0, nullable=False)

    # Relationships
    user = relationship("User", back_populates="progress")
    lesson = relationship("Lesson", back_populates="progress")

    def __repr__(self) -> str:
        return f"<UserProgress(user_id={self.user_id}, lesson_id={self.lesson_id}, completed={self.completed_at is not None})>"
