"""Content service for managing textbook content and indexing."""
import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.lesson import Lesson
from src.services.chatbot import chatbot_service

logger = logging.getLogger(__name__)

class ContentService:
    """Service class for handling textbook content operations."""

    def __init__(self):
        self.chatbot_service = chatbot_service

    async def index_new_lesson(self, db: AsyncSession, lesson_id: UUID) -> bool:
        """
        Index a new lesson in the vector database for RAG search.

        Args:
            db: Database session
            lesson_id: ID of the lesson to index

        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            # Get the lesson from the database
            from src.crud.lesson import get_lesson
            lesson = await get_lesson(db, lesson_id)

            if not lesson:
                logger.error(f"Lesson with ID {lesson_id} not found for indexing")
                return False

            # Index the lesson content using the chatbot service
            indexed_count = await self.chatbot_service.index_lesson_content(db, lesson)

            logger.info(f"Successfully indexed lesson {lesson_id} with {indexed_count} content chunks")
            return True

        except Exception as e:
            logger.error(f"Error indexing lesson {lesson_id}: {e}")
            return False

    async def update_lesson_index(self, db: AsyncSession, lesson_id: UUID) -> bool:
        """
        Update the index for an existing lesson.

        This removes the old index and creates a new one.

        Args:
            db: Database session
            lesson_id: ID of the lesson to update in index

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # First, remove the old lesson from the index
            await self.chatbot_service.delete_lesson_from_index(db, lesson_id)

            # Then index the updated lesson
            success = await self.index_new_lesson(db, lesson_id)

            if success:
                logger.info(f"Successfully updated index for lesson {lesson_id}")
            else:
                logger.error(f"Failed to update index for lesson {lesson_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating index for lesson {lesson_id}: {e}")
            return False

    async def remove_lesson_from_index(self, db: AsyncSession, lesson_id: UUID) -> bool:
        """
        Remove a lesson from the vector database index.

        Args:
            db: Database session
            lesson_id: ID of the lesson to remove from index

        Returns:
            bool: True if removal was successful, False otherwise
        """
        try:
            success = await self.chatbot_service.delete_lesson_from_index(db, lesson_id)

            if success:
                logger.info(f"Successfully removed lesson {lesson_id} from index")
            else:
                logger.error(f"Failed to remove lesson {lesson_id} from index")

            return success

        except Exception as e:
            logger.error(f"Error removing lesson {lesson_id} from index: {e}")
            return False

# Global instance
content_service = ContentService()