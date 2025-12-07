"""Chatbot service for RAG operations using Qdrant and LLM integration."""
import time
import logging
from typing import List, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.llm_integrations import generate_embeddings, generate_llm_response
from src.services.qdrant_client import get_qdrant_service
from src.models.lesson import Lesson
from src.crud.lesson_embedding import (
    create_lesson_embedding,
    delete_lesson_embeddings_by_lesson
)

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service class for handling RAG chatbot operations."""

    def __init__(self):
        self.qdrant_service = get_qdrant_service()

    async def index_lesson_content(self, db: AsyncSession, lesson: Lesson) -> int:
        """
        Index lesson content in Qdrant for RAG search.

        This method chunks the lesson content, generates embeddings,
        stores them in Qdrant, and creates references in PostgreSQL.

        Returns:
            Number of indexed chunks
        """
        # Chunk lesson content
        content_chunks = self._chunk_content(lesson.content_markdown)

        # Process code examples as separate chunks
        for code_example in lesson.code_examples:
            content_chunks.append(f"Code Example ({code_example.language}): {code_example.code}")

        indexed_count = 0

        for chunk in content_chunks:
            # Generate embedding for the chunk
            embeddings = generate_embeddings([chunk])
            embedding_vector = embeddings[0]

            # Store in Qdrant
            import uuid
            qdrant_vector_id = str(uuid.uuid4())  # Using UUID as Qdrant point ID

            # Prepare payload with lesson info
            payload = {
                "lesson_id": str(lesson.lesson_id),
                "module_id": str(lesson.module_id),
                "title": lesson.title,
                "passage_text": chunk
            }

            # Upsert to Qdrant
            self.qdrant_service.client.upsert(
                collection_name=self.qdrant_service.collection_name,
                points=[
                    {
                        "id": qdrant_vector_id,
                        "vector": embedding_vector,
                        "payload": payload
                    }
                ]
            )

            # Create reference in PostgreSQL
            embedding_data = {
                "lesson_id": lesson.lesson_id,
                "passage_text": chunk,
                "qdrant_vector_id": qdrant_vector_id
            }

            await create_lesson_embedding(db, embedding_data)
            indexed_count += 1

        return indexed_count

    async def delete_lesson_from_index(self, db: AsyncSession, lesson_id: UUID) -> bool:
        """Remove lesson content from Qdrant index."""
        # Delete from Qdrant first
        qdrant_result = self.qdrant_service.delete_embeddings_by_lesson(lesson_id)

        # Then delete references from PostgreSQL
        embedding_count = await delete_lesson_embeddings_by_lesson(db, lesson_id)

        if qdrant_result and embedding_count > 0:
            logger.info(f"Removed {embedding_count} embeddings from both Qdrant and PostgreSQL for lesson {lesson_id}")
        elif embedding_count > 0:
            logger.warning(f"Removed {embedding_count} embeddings from PostgreSQL but Qdrant deletion may have failed for lesson {lesson_id}")

        return qdrant_result

    async def search_relevant_passages(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant passages in Qdrant based on the query.

        Returns:
            List of dictionaries with lesson_id, passage_text, and similarity_score
        """
        # Generate embedding for the query
        query_embeddings = generate_embeddings([query_text])
        query_vector = query_embeddings[0]

        # Search in Qdrant
        search_results = self.qdrant_service.client.search(
            collection_name=self.qdrant_service.collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=similarity_threshold
        )

        # Format results
        results = []
        for result in search_results:
            if result.score >= similarity_threshold:
                results.append({
                    "lesson_id": UUID(result.payload["lesson_id"]),
                    "passage_text": result.payload["passage_text"],
                    "similarity_score": result.score,
                    "title": result.payload.get("title", ""),
                    "module_id": UUID(result.payload.get("module_id", ""))
                })

        return results

    async def generate_response(
        self,
        query_text: str,
        retrieved_passages: List[Dict[str, Any]],
        model_name: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM based on retrieved passages.

        Returns:
            Dictionary with response_text and citations
        """
        start_time = time.time()

        # Generate response using LLM
        llm_result = generate_llm_response(
            query_text=query_text,
            retrieved_passages=retrieved_passages,
            model_name=model_name
        )

        response_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

        return {
            "response_text": llm_result["response_text"],
            "citations": llm_result["citations"],
            "response_generation_time_ms": response_time
        }

    def _chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk content into smaller pieces for embedding.

        Args:
            content: The text content to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of content chunks
        """
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # If we're near the end, include the rest
            if end >= len(content):
                chunks.append(content[start:])
                break

            # Try to break at sentence boundary
            chunk = content[start:end]

            # Find the last sentence boundary within the chunk
            last_period = chunk.rfind('.')
            last_exclamation = chunk.rfind('!')
            last_question = chunk.rfind('?')
            last_boundary = max(last_period, last_exclamation, last_question)

            if last_boundary > chunk_size // 2:  # Only split if the boundary is reasonably far in
                actual_end = start + last_boundary + 1
                chunks.append(content[start:actual_end])
                start = actual_end - overlap
            else:
                # If no good boundary found, just take the chunk and advance by chunk_size
                chunks.append(content[start:end])
                start = end - overlap

        return chunks

# Global instance
chatbot_service = ChatbotService()