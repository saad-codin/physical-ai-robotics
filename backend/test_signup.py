"""Test script to debug signup endpoint."""
import asyncio
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.session import async_session
from src.schemas.user import UserCreate
from src.crud import user as user_crud
from src.core.security import get_password_hash

async def test_user_creation():
    """Test creating a user directly."""
    try:
        async with async_session() as db:
            # Create test user data
            user_in = UserCreate(
                email="directtest@example.com",
                password="password123",
                ros_experience_level="beginner",
                focus_area="both"
            )
            
            # Check if user exists
            existing = await user_crud.get_user_by_email(db, email=user_in.email)
            if existing:
                print(f"User already exists: {existing.email}")
                return
            
            # Prepare user data
            hashed_password = get_password_hash(user_in.password)
            user_in_db = user_in.model_dump()
            user_in_db["password_hash"] = hashed_password
            del user_in_db["password"]
            
            # Handle Enum values
            user_data = {
                **user_in_db,
                "specialization": user_in.specialization or [],
                "ros_experience_level": user_in.ros_experience_level.value if hasattr(user_in.ros_experience_level, "value") else user_in.ros_experience_level,
                "focus_area": user_in.focus_area.value if hasattr(user_in.focus_area, "value") else user_in.focus_area,
                "language_preference": user_in.language_preference,
                "is_active": True
            }
            
            print(f"Creating user with data: {user_data}")
            
            # Create user
            new_user = await user_crud.create_user(db, user=user_data)
            print(f"✓ User created successfully: {new_user.email} (ID: {new_user.user_id})")
            
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_user_creation())
