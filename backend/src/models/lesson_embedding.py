"""LessonEmbedding model for vector storage in Qdrant."""
from sqlalchemy import Column, Text, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base, TimestampMixin


class LessonEmbedding(Base, TimestampMixin):
    """LessonEmbedding entity for semantic search in RAG chatbot.

    Stores:
    - Chunked passages from lesson content
    - Qdrant vector ID reference (actual embeddings stored in Qdrant)
    - Mapping between Qdrant vector ID and lesson content
    """
    __tablename__ = "lesson_embeddings"

    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("lessons.lesson_id"), nullable=False, index=True)
    passage_text = Column(Text, nullable=False)
    qdrant_vector_id = Column(String(255), nullable=False, unique=True)  # Reference to Qdrant vector ID

    # Relationships
    lesson = relationship("Lesson", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<LessonEmbedding(embedding_id={self.embedding_id}, lesson_id={self.lesson_id}, qdrant_vector_id={self.qdrant_vector_id})>"
