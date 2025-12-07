"""Personalization service for content recommendation and filtering."""
import logging
from typing import List, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User
from src.models.lesson import Lesson
from src.models.module import Module
from src.models.user_progress import UserProgress
from src.crud.user import get_user
from src.crud.lesson import get_lessons
from src.crud.user_progress import get_user_progresses
from src.crud.module import get_modules

logger = logging.getLogger(__name__)

class PersonalizationService:
    """Service class for handling content personalization based on user profiles and progress."""

    async def get_personalized_lesson_recommendations(
        self,
        db: AsyncSession,
        user_id: UUID,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get personalized lesson recommendations based on user profile and progress.

        Args:
            db: Database session
            user_id: ID of the user to get recommendations for
            limit: Maximum number of recommendations to return

        Returns:
            List of recommended lessons with personalization metadata
        """
        user = await get_user(db, user_id)
        if not user:
            logger.warning(f"User {user_id} not found for personalization")
            return []

        # Get user's progress to identify completed lessons
        user_progresses = await get_user_progresses(db, user_id=user_id)
        completed_lesson_ids = {str(progress.lesson_id) for progress in user_progresses if progress.completed_at}

        # Get all lessons
        all_lessons = await get_lessons(db)

        # Filter out completed lessons
        available_lessons = [lesson for lesson in all_lessons if str(lesson.lesson_id) not in completed_lesson_ids]

        # Apply personalization logic based on user profile
        personalized_lessons = []
        for lesson in available_lessons:
            score = self._calculate_personalization_score(user, lesson)
            if score > 0:  # Only include lessons with positive score
                personalized_lessons.append({
                    "lesson_id": str(lesson.lesson_id),
                    "title": lesson.title,
                    "module_name": lesson.module.name.value,
                    "learning_objectives": lesson.learning_objectives,
                    "personalization_score": score,
                    "relevance_reasons": self._get_relevance_reasons(user, lesson)
                })

        # Sort by personalization score (descending) and return top N
        personalized_lessons.sort(key=lambda x: x["personalization_score"], reverse=True)
        return personalized_lessons[:limit]

    async def get_personalized_learning_path(
        self,
        db: AsyncSession,
        user_id: UUID
    ) -> Dict[str, Any]:
        """
        Get a personalized learning path for the user based on their profile and progress.

        Args:
            db: Database session
            user_id: ID of the user to get learning path for

        Returns:
            Dictionary containing personalized learning path information
        """
        user = await get_user(db, user_id)
        if not user:
            logger.warning(f"User {user_id} not found for learning path")
            return {"modules": [], "recommended_next": None}

        # Get user's progress
        user_progresses = await get_user_progresses(db, user_id=user_id)
        completed_lesson_ids = {str(progress.lesson_id) for progress in user_progresses if progress.completed_at}

        # Get all modules
        modules = await get_modules(db)

        # Build personalized path
        path_modules = []
        next_recommendation = None

        for module in modules:
            lessons = await get_lessons(db, module_id=module.module_id)

            # Calculate module progress
            module_lessons = [lesson for lesson in lessons]
            completed_in_module = [lesson for lesson in module_lessons if str(lesson.lesson_id) in completed_lesson_ids]
            module_progress = len(completed_in_module) / len(module_lessons) if module_lessons else 0

            # Find next lesson in this module if not fully completed
            next_lesson = None
            if len(completed_in_module) < len(module_lessons):
                # Find first incomplete lesson in order
                for lesson in sorted(module_lessons, key=lambda x: x.order_index):
                    if str(lesson.lesson_id) not in completed_lesson_ids:
                        next_lesson = lesson
                        break

            path_modules.append({
                "module_id": str(module.module_id),
                "name": module.name.value,
                "description": module.description,
                "progress": module_progress,
                "total_lessons": len(module_lessons),
                "completed_lessons": len(completed_in_module),
                "next_lesson": {
                    "lesson_id": str(next_lesson.lesson_id),
                    "title": next_lesson.title
                } if next_lesson else None
            })

            # Set overall next recommendation if not already set and this module has next lesson
            if not next_recommendation and next_lesson:
                next_recommendation = {
                    "lesson_id": str(next_lesson.lesson_id),
                    "title": next_lesson.title,
                    "module_name": module.name.value,
                    "personalization_score": self._calculate_personalization_score(user, next_lesson),
                    "relevance_reasons": self._get_relevance_reasons(user, next_lesson)
                }

        return {
            "modules": path_modules,
            "recommended_next": next_recommendation
        }

    def _calculate_personalization_score(self, user: User, lesson: Lesson) -> float:
        """
        Calculate personalization score for a lesson based on user profile.

        Args:
            user: User object with profile information
            lesson: Lesson object to score

        Returns:
            Score between 0 and 1 (higher is more personalized)
        """
        score = 0.0

        # Base score
        score += 0.1  # Every lesson gets a base score

        # Specialization match
        if lesson.module.name.value in user.specialization:
            score += 0.3

        # Focus area match (if lesson is hardware or software related)
        lesson_content_keywords = lesson.content_markdown.lower()
        if user.focus_area == "HARDWARE" and any(keyword in lesson_content_keywords for keyword in ["hardware", "sensor", "motor", "actuator", "embedded"]):
            score += 0.2
        elif user.focus_area == "SOFTWARE" and any(keyword in lesson_content_keywords for keyword in ["algorithm", "code", "software", "programming", "architecture"]):
            score += 0.2

        # Experience level match (simplified)
        if user.ros_experience_level == "BEGINNER":
            # Beginner-friendly lessons get higher score
            if "beginner" in lesson.title.lower() or "introduction" in lesson.title.lower():
                score += 0.15
        elif user.ros_experience_level == "ADVANCED":
            # Advanced lessons get higher score for advanced users
            if "advanced" in lesson.title.lower() or "expert" in lesson.title.lower():
                score += 0.15

        # Language preference match
        if user.language_preference != "en":
            # For non-English users, prefer translated content (when available)
            # This would be enhanced when translation features are implemented
            score += 0.05

        # Cap the score at 1.0
        return min(score, 1.0)

    def _get_relevance_reasons(self, user: User, lesson: Lesson) -> List[str]:
        """
        Get reasons why a lesson is relevant to the user.

        Args:
            user: User object with profile information
            lesson: Lesson object

        Returns:
            List of reasons for relevance
        """
        reasons = []

        if lesson.module.name.value in user.specialization:
            reasons.append(f"Matches your specialization in '{lesson.module.name.value}'")

        lesson_content_keywords = lesson.content_markdown.lower()
        if user.focus_area == "HARDWARE" and any(keyword in lesson_content_keywords for keyword in ["hardware", "sensor", "motor", "actuator", "embedded"]):
            reasons.append(f"Matches your focus area in '{user.focus_area.lower()}'")
        elif user.focus_area == "SOFTWARE" and any(keyword in lesson_content_keywords for keyword in ["algorithm", "code", "software", "programming", "architecture"]):
            reasons.append(f"Matches your focus area in '{user.focus_area.lower()}'")

        if user.ros_experience_level == "BEGINNER" and any(keyword in lesson.title.lower() for keyword in ["beginner", "introduction", "fundamentals"]):
            reasons.append("Appropriate for your experience level")
        elif user.ros_experience_level == "ADVANCED" and any(keyword in lesson.title.lower() for keyword in ["advanced", "expert", "complex"]):
            reasons.append("Appropriate for your experience level")

        return reasons

# Global instance
personalization_service = PersonalizationService()