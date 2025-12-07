"""Personalization API endpoints for content recommendation and filtering."""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.personalization import (
    LessonRecommendationsResponse,
    PersonalizedLearningPathResponse
)
from src.models.user import User as DBUser
from src.core.auth import get_current_active_user
from src.services.personalization import personalization_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/recommendations", response_model=LessonRecommendationsResponse)
async def get_personalized_recommendations(
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get personalized lesson recommendations based on user profile and progress."""
    try:
        recommendations = await personalization_service.get_personalized_lesson_recommendations(
            db, current_user.user_id
        )

        # Format recommendations to match the schema
        formatted_recommendations = []
        for rec in recommendations:
            formatted_recommendations.append({
                "lesson_id": rec["lesson_id"],
                "title": rec["title"],
                "module_name": rec["module_name"],
                "personalization_score": rec["personalization_score"],
                "relevance_reasons": rec["relevance_reasons"]
            })

        return {"recommendations": formatted_recommendations}

    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting personalized recommendations"
        )


@router.get("/learning-path", response_model=PersonalizedLearningPathResponse)
async def get_personalized_learning_path(
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a personalized learning path for the user."""
    try:
        learning_path = await personalization_service.get_personalized_learning_path(
            db, current_user.user_id
        )

        return learning_path

    except Exception as e:
        logger.error(f"Error getting personalized learning path: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting personalized learning path"
        )


@router.get("/profile-analysis")
async def get_profile_analysis(
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get an analysis of the user's profile and how it affects personalization."""
    try:
        # This endpoint provides insights into how the user's profile is being used
        profile_analysis = {
            "user_id": str(current_user.user_id),
            "specialization": current_user.specialization,
            "ros_experience_level": current_user.ros_experience_level.value,
            "focus_area": current_user.focus_area.value,
            "language_preference": current_user.language_preference,
            "profile_used_for": [
                "Content recommendations",
                "Learning path personalization",
                "Module sequence suggestions",
                "Difficulty level matching"
            ]
        }

        return profile_analysis

    except Exception as e:
        logger.error(f"Error getting profile analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting profile analysis"
        )