from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.module import ModuleRead, ModuleCreate, ModuleUpdate
from src.crud import module as module_crud
from src.models.module import Module as DBModule

router = APIRouter()

@router.post("/", response_model=ModuleRead, status_code=status.HTTP_201_CREATED)
async def create_module(
    module_in: ModuleCreate,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    module = await module_crud.get_module_by_name(db, name=module_in.name)
    if module:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Module with this name already exists"
        )
    return await module_crud.create_module(db, module=module_in)

@router.get("/{module_id}", response_model=ModuleRead)
async def get_module_by_id(
    module_id: str,
    db: AsyncSession = Depends(get_db)
):
    module = await module_crud.get_module(db, module_id=module_id)
    if not module:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Module not found")
    return module

@router.get("/", response_model=List[ModuleRead])
async def get_all_modules(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    modules = await module_crud.get_modules(db, skip=skip, limit=limit)
    return modules

@router.put("/{module_id}", response_model=ModuleRead)
async def update_module(
    module_id: str,
    module_in: ModuleUpdate,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    module = await module_crud.get_module(db, module_id=module_id)
    if not module:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Module not found")
    return await module_crud.update_module(db, db_module=module, module_in=module_in)

@router.delete("/{module_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_module(
    module_id: str,
    db: AsyncSession = Depends(get_db)
):
    # This endpoint might be admin-only in a real application
    module = await module_crud.get_module(db, module_id=module_id)
    if not module:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Module not found")
    await module_crud.delete_module(db, module_id=module_id)
    return None
