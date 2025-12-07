"""ContentTranslation model for multi-language support."""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base, TimestampMixin


class ContentTranslation(Base, TimestampMixin):
    """ContentTranslation entity for multi-language textbook content.

    Constitution Principle: Supports global accessibility through professional translations.

    Stores:
    - Translated lesson titles and content
    - Language codes (ISO 639-1: en, zh, es, fr, etc.)
    - Review status for quality assurance
    """
    __tablename__ = "content_translations"

    translation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("lessons.lesson_id"), nullable=False, index=True)
    language_code = Column(String(10), nullable=False, index=True)  # ISO 639-1 language codes
    translated_title = Column(String(500), nullable=False)
    translated_content_markdown = Column(Text, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)  # Null if not yet reviewed

    # Relationships
    lesson = relationship("Lesson", back_populates="translations")

    def __repr__(self) -> str:
        return f"<ContentTranslation(translation_id={self.translation_id}, lesson_id={self.lesson_id}, language={self.language_code})>"
