import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.lesson import LessonRead, LessonCreate, LessonUpdate
from src.crud import lesson as lesson_crud
from src.models.lesson import Lesson as DBLesson
from src.services.content import content_service
from src.services.translation import translation_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=LessonRead, status_code=status.HTTP_201_CREATED)
async def create_lesson(
    lesson_in: LessonCreate,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    created_lesson = await lesson_crud.create_lesson(db, lesson=lesson_in)

    # Index the lesson content for RAG search
    indexing_success = await content_service.index_new_lesson(db, created_lesson.lesson_id)

    if not indexing_success:
        logger.warning(f"Failed to index lesson {created_lesson.lesson_id} for RAG search")

    return created_lesson

@router.get("/{lesson_id}", response_model=LessonRead)
async def get_lesson_by_id(
    lesson_id: str,
    language: str = "en",  # Default to English
    db: AsyncSession = Depends(get_db)
):
    lesson = await lesson_crud.get_lesson(db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")

    # If a specific language is requested and it's not the default, get the translation
    if language != "en":
        try:
            from src.models.user import User as DBUser
            from src.core.auth import get_current_active_user
            # Note: In a real implementation, you might want to pass the user for personalization
            # For now, we'll call the translation service without a user
            translated_content = await translation_service.get_translated_content(
                db,
                UUID(lesson_id),
                language,
                None  # No user context for now
            )

            # Update the lesson content with the translated version
            lesson.title = translated_content["title"]
            lesson.content_markdown = translated_content["content"]
        except Exception as e:
            logger.warning(f"Could not translate lesson {lesson_id} to {language}: {e}")
            # Fall back to original content if translation fails

    return lesson


@router.get("/", response_model=List[LessonRead])
async def get_all_lessons(
    module_id: str = None, # Optional filter by module
    language: str = "en",  # Default to English
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    lessons = await lesson_crud.get_lessons(db, module_id=module_id, skip=skip, limit=limit)

    # If a specific language is requested, get translations for all lessons
    if language != "en":
        try:
            for lesson in lessons:
                translated_content = await translation_service.get_translated_content(
                    db,
                    lesson.lesson_id,
                    language,
                    None  # No user context for now
                )

                lesson.title = translated_content["title"]
                lesson.content_markdown = translated_content["content"]
        except Exception as e:
            logger.warning(f"Could not translate lessons to {language}: {e}")
            # Fall back to original content if translation fails

    return lessons


@router.put("/{lesson_id}", response_model=LessonRead)
async def update_lesson(
    lesson_id: str,
    lesson_in: LessonUpdate,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    lesson = await lesson_crud.get_lesson(db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")

    updated_lesson = await lesson_crud.update_lesson(db, db_lesson=lesson, lesson_in=lesson_in)

    # Update the lesson index for RAG search
    indexing_success = await content_service.update_lesson_index(db, updated_lesson.lesson_id)

    if not indexing_success:
        logger.warning(f"Failed to update index for lesson {updated_lesson.lesson_id}")

    return updated_lesson

@router.delete("/{lesson_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lesson(
    lesson_id: str,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    lesson = await lesson_crud.get_lesson(db, lesson_id=lesson_id)
    if not lesson:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")

    # Remove the lesson from the index before deleting from DB
    indexing_success = await content_service.remove_lesson_from_index(db, lesson.lesson_id)

    if not indexing_success:
        logger.warning(f"Failed to remove lesson {lesson.lesson_id} from index")

    await lesson_crud.delete_lesson(db, lesson_id=lesson_id)
    return None
