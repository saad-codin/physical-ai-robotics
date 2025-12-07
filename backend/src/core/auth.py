"""Authentication utilities for JWT token handling and user validation."""
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.user import User
from src.db.session import get_db
from src.core.security import verify_token

# HTTP Bearer token security scheme
security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token.

    Args:
        data: Data to encode in the token (typically user ID)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default 15 minutes

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.auth_secret, algorithm="HS256")
    return encoded_jwt


async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency to get the current authenticated user.

    Validates JWT token and retrieves user from database.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        User: The authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials

    user_id = verify_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Retrieve user from database
    from src.crud.user import get_user
    user = await get_user(db, UUID(user_id))

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user

optional_security = HTTPBearer(auto_error=False)


async def get_optional_current_active_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Dependency to optionally get the current authenticated user.

    If no token is provided, returns None. If a token is provided, it is
    validated, and the user is retrieved from the database.

    Args:
        credentials: Optional HTTP Bearer token credentials.
        db: Database session.

    Returns:
        Optional[User]: The authenticated user or None.
    """
    if credentials is None:
        return None

    token = credentials.credentials
    try:
        user_id = verify_token(token)
        if user_id is None:
            return None

        from src.crud.user import get_user
        user = await get_user(db, UUID(user_id))

        if user and user.is_active:
            return user
        return None
    except Exception:
        return None
