"""ChatbotQuery model for logging RAG chatbot interactions."""
from sqlalchemy import Column, Text, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from src.db.base import Base, TimestampMixin


class ChatbotQuery(Base, TimestampMixin):
    """ChatbotQuery entity logging chatbot interactions for analytics.

    Tracks:
    - User queries
    - Retrieved passages from Qdrant
    - Generated responses
    - Response generation time
    - Citations
    """
    __tablename__ = "chatbot_queries"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True, index=True)
    query_text = Column(Text, nullable=False)
    retrieved_passages = Column(JSONB, nullable=False, default=list)  # Array of {lesson_id, passage_text, similarity_score}
    response_text = Column(Text, nullable=False)
    response_generation_time_ms = Column(Integer, nullable=False)

    # Relationships
    user = relationship("User", back_populates="chatbot_queries")

    def __repr__(self) -> str:
        return f"<ChatbotQuery(query_id={self.query_id}, user_id={self.user_id})>"
