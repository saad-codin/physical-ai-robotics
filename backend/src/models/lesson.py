"""Lesson model for textbook content."""
from sqlalchemy import Column, String, Integer, Text, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base, TimestampMixin


class Lesson(Base, TimestampMixin):
    """Lesson entity representing a single chapter/unit within a module.

    Constitution Principles:
    - Rigor & Accuracy: All content must cite official documentation
    - Academic Clarity: Progressive learning from fundamentals to advanced
    - Reproducibility: Code examples must be copy-paste ready
    """
    __tablename__ = "lessons"

    lesson_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_id = Column(UUID(as_uuid=True), ForeignKey("modules.module_id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    learning_objectives = Column(ARRAY(Text), nullable=False, default=list)
    content_markdown = Column(Text, nullable=False)
    code_examples = Column(JSONB, nullable=False, default=list)  # Array of {language, code, description}
    diagrams = Column(ARRAY(String), nullable=False, default=list)  # Array of image URLs/paths
    order_index = Column(Integer, nullable=False)

    # Relationships
    module = relationship("Module", back_populates="lessons")
    progress = relationship("UserProgress", back_populates="lesson", cascade="all, delete-orphan")
    embeddings = relationship("LessonEmbedding", back_populates="lesson", cascade="all, delete-orphan")
    translations = relationship("ContentTranslation", back_populates="lesson", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Lesson(lesson_id={self.lesson_id}, title={self.title})>"
