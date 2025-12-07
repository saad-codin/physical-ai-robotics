from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.user_progress import UserProgressRead, UserProgressCreate, UserProgressUpdate
from src.crud import user_progress as user_progress_crud
from src.models.user_progress import UserProgress as DBUserProgress
from src.models.user import User as DBUser
from src.core.auth import get_current_active_user

router = APIRouter()

@router.post("/", response_model=UserProgressRead, status_code=status.HTTP_201_CREATED)
async def create_user_progress(
    user_progress_in: UserProgressCreate,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    # Ensure the progress is for the current user
    if user_progress_in.user_id and str(user_progress_in.user_id) != str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create progress for another user"
        )
    user_progress_in.user_id = current_user.user_id # Ensure correct user_id

    existing_progress = await user_progress_crud.get_user_progress_by_user_and_lesson(
        db, user_id=current_user.user_id, lesson_id=user_progress_in.lesson_id
    )
    if existing_progress:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User progress for this lesson already exists"
        )
    return await user_progress_crud.create_user_progress(db, user_progress=user_progress_in)

@router.get("/me", response_model=List[UserProgressRead])
async def get_my_progress(
    lesson_id: Optional[str] = None,
    bookmarked: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    progress = await user_progress_crud.get_user_progresses(
        db, user_id=current_user.user_id, lesson_id=lesson_id, bookmarked=bookmarked, skip=skip, limit=limit
    )
    return progress

@router.get("/{progress_id}", response_model=UserProgressRead)
async def get_user_progress_by_id(
    progress_id: str,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    progress = await user_progress_crud.get_user_progress(db, progress_id=progress_id)
    if not progress:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User progress not found")
    if str(progress.user_id) != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this progress")
    return progress

@router.put("/{progress_id}", response_model=UserProgressRead)
async def update_user_progress(
    progress_id: str,
    user_progress_in: UserProgressUpdate,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    progress = await user_progress_crud.get_user_progress(db, progress_id=progress_id)
    if not progress:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User progress not found")
    if str(progress.user_id) != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to update this progress")

    # Ensure user_id cannot be changed
    if user_progress_in.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID cannot be changed in progress update"
        )

    return await user_progress_crud.update_user_progress(db, db_user_progress=progress, user_progress_in=user_progress_in)

@router.delete("/{progress_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_progress(
    progress_id: str,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    progress = await user_progress_crud.get_user_progress(db, progress_id=progress_id)
    if not progress:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User progress not found")
    if str(progress.user_id) != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this progress")

    await user_progress_crud.delete_user_progress(db, progress_id=progress_id)
    return None
