"""Security utilities for password hashing and token verification."""
from datetime import datetime
from typing import Optional
import bcrypt
import jwt

from src.config import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password.

    Args:
        plain_password: The plain text password
        hashed_password: The hashed password to verify against

    Returns:
        bool: True if password matches, False otherwise
    """
    # Bcrypt handles bytes, and has a 72-byte limit
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: The plain text password

    Returns:
        str: The hashed password
    """
    # Bcrypt has a 72-byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_token(token: str) -> Optional[str]:
    """Verify a JWT token and return the user ID.

    Args:
        token: The JWT token to verify

    Returns:
        Optional[str]: The user ID if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.auth_secret,
            algorithms=["HS256"]
        )
        user_id: str = payload.get("sub")
        return user_id
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None