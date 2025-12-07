"""Qdrant client for vector database operations."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for managing Qdrant vector database operations.

    Constitution Principle: RAG chatbot must use semantic search
    via Qdrant for retrieving relevant lesson passages.
    """

    def __init__(self):
        """Initialize Qdrant client and ensure collection exists."""
        # Support in-memory mode for development
        if settings.qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
            logger.info("Using Qdrant in-memory mode")
        else:
            # Check if this is a Qdrant Cloud URL (contains cloud.qdrant.io)
            if "cloud.qdrant.io" in settings.qdrant_url:
                # For Qdrant Cloud, extract the host and use HTTPS
                # URL format: https://<id>.<region>.cloud.qdrant.io:6333
                import re
                # Extract the host part from the full URL
                host_match = re.search(r'https://([^/:]+)', settings.qdrant_url)
                if host_match:
                    host = host_match.group(1)
                    self.client = QdrantClient(
                        host=host,
                        api_key=settings.qdrant_api_key,
                        https=True,
                        timeout=10.0,  # Add timeout to prevent hanging
                    )
                    logger.info(f"Connected to Qdrant Cloud at {host}")
                else:
                    # Fallback to original method if regex doesn't match
                    self.client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=10.0,
                    )
                    logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
            else:
                # For local or self-hosted instances, use the original method
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    timeout=10.0,  # Add timeout to prevent hanging
                )
                logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
        
        self.collection_name = settings.qdrant_collection_name
        self.vector_dimension = settings.qdrant_vector_dimension
        # Defer collection creation to first use to avoid startup issues
        self._collection_ensured = False

    def _ensure_collection_exists(self):
        """Create the collection if it doesn't exist."""
        if self._collection_ensured:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection already exists: {self.collection_name}")
            self._collection_ensured = True
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {e}")
            # Don't raise here to allow application to start, just log the error
            # The collection will be created when first needed

    def upsert_embedding(
        self,
        embedding_id: UUID,
        lesson_id: UUID,
        passage_text: str,
        embedding_vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert a lesson embedding into Qdrant.

        Args:
            embedding_id: Unique ID for the embedding
            lesson_id: ID of the lesson this embedding belongs to
            passage_text: The text passage
            embedding_vector: The embedding vector (1536 dimensions)
            metadata: Optional additional metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure collection exists before operation
            self._ensure_collection_exists()

            payload = {
                "embedding_id": str(embedding_id),
                "lesson_id": str(lesson_id),
                "passage_text": passage_text,
                **(metadata or {}),
            }

            point = PointStruct(
                id=str(embedding_id),
                vector=embedding_vector,
                payload=payload,
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info(f"Upserted embedding {embedding_id} for lesson {lesson_id}")
            return True

        except Exception as e:
            logger.error(f"Error upserting embedding: {e}")
            return False

    def search_similar_passages(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Search for similar passages using vector similarity.

        Args:
            query_vector: The query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List[Dict[str, Any]]: List of similar passages with metadata
        """
        try:
            # Ensure collection exists before operation
            self._ensure_collection_exists()

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )

            results = []
            for scored_point in search_result:
                results.append({
                    "embedding_id": scored_point.id,
                    "lesson_id": scored_point.payload.get("lesson_id"),
                    "passage_text": scored_point.payload.get("passage_text"),
                    "similarity_score": scored_point.score,
                    "metadata": {
                        k: v for k, v in scored_point.payload.items()
                        if k not in ["embedding_id", "lesson_id", "passage_text"]
                    },
                })

            logger.info(f"Found {len(results)} similar passages")
            return results

        except Exception as e:
            logger.error(f"Error searching similar passages: {e}")
            return []

    def delete_embeddings_by_lesson(self, lesson_id: UUID) -> bool:
        """Delete all embeddings for a specific lesson.

        Args:
            lesson_id: The lesson ID to delete embeddings for

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure collection exists before operation
            self._ensure_collection_exists()

            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "lesson_id",
                                "match": {"value": str(lesson_id)},
                            }
                        ]
                    }
                },
            )

            logger.info(f"Deleted embeddings for lesson {lesson_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection.

        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            # Ensure collection exists before operation
            self._ensure_collection_exists()

            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,  # Use the stored collection name
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "points_count": collection_info.points_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


# Singleton instance (lazy initialization to avoid startup issues)
_qdrant_service = None


def get_qdrant_service():
    """Get the Qdrant service instance, creating it if it doesn't exist."""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
