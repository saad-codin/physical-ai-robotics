from typing import Optional, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.models.module import Module, ModuleName
from src.schemas.module import ModuleCreate, ModuleUpdate


async def get_module(db: AsyncSession, module_id: UUID) -> Optional[Module]:
    """Get a module by ID with lessons loaded."""
    stmt = select(Module).where(Module.module_id == module_id).options(
        selectinload(Module.lessons)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_module_by_name(db: AsyncSession, name: ModuleName) -> Optional[Module]:
    """Get a module by name."""
    stmt = select(Module).where(Module.name == name)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_modules(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Module]:
    """Get a list of modules."""
    stmt = select(Module).order_by(Module.order_index).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()


async def create_module(db: AsyncSession, module: ModuleCreate) -> Module:
    """Create a new module."""
    db_module = Module(**module.model_dump())
    db.add(db_module)
    await db.commit()
    await db.refresh(db_module)
    return db_module


async def update_module(db: AsyncSession, db_module: Module, module_in: ModuleUpdate) -> Module:
    """Update a module."""
    update_data = module_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_module, field, value)
    await db.commit()
    await db.refresh(db_module)
    return db_module


async def delete_module(db: AsyncSession, module_id: UUID) -> bool:
    """Delete a module."""
    stmt = select(Module).where(Module.module_id == module_id)
    result = await db.execute(stmt)
    db_module = result.scalar_one_or_none()
    if db_module:
        await db.delete(db_module)
        await db.commit()
        return True
    return False