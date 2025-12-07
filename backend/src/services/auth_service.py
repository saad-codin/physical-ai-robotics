"""Authentication service using Better-Auth patterns."""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from passlib.context import CryptContext
import jwt
from sqlalchemy.orm import Session
from uuid import UUID

from src.config import settings
from src.models.user import User
from src.schemas.user import UserCreate


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service for user management and token generation."""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password.

        Args:
            plain_password: The plain text password
            hashed_password: The hashed password to verify against

        Returns:
            bool: True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: The plain text password

        Returns:
            str: The hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(user_id: UUID, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token.

        Args:
            user_id: The user's UUID
            expires_delta: Optional custom expiration time

        Returns:
            str: The encoded JWT token
        """
        if expires_delta is None:
            expires_delta = timedelta(hours=settings.auth_token_expiry_hours)

        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "type": "access"
        }

        encoded_jwt = jwt.encode(
            to_encode,
            settings.auth_secret,
            algorithm="HS256"
        )
        return encoded_jwt

    @staticmethod
    def create_refresh_token(user_id: UUID) -> str:
        """Create a JWT refresh token.

        Args:
            user_id: The user's UUID

        Returns:
            str: The encoded JWT refresh token
        """
        expires_delta = timedelta(days=settings.auth_refresh_token_expiry_days)
        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "type": "refresh"
        }

        encoded_jwt = jwt.encode(
            to_encode,
            settings.auth_secret,
            algorithm="HS256"
        )
        return encoded_jwt

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[str]:
        """Verify a JWT token and return the user ID.

        Args:
            token: The JWT token to verify
            token_type: Expected token type ("access" or "refresh")

        Returns:
            Optional[str]: The user ID if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                settings.auth_secret,
                algorithms=["HS256"]
            )

            if payload.get("type") != token_type:
                return None

            user_id: str = payload.get("sub")
            return user_id
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None

    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password.

        Args:
            db: Database session
            email: User's email
            password: User's plain text password

        Returns:
            Optional[User]: The authenticated user or None if authentication fails
        """
        user = db.query(User).filter(User.email == email).first()

        if not user:
            return None

        if not user.password_hash:
            # OAuth user without password
            return None

        if not AuthService.verify_password(password, user.password_hash):
            return None

        return user

    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """Create a new user.

        Args:
            db: Database session
            user_data: User creation data

        Returns:
            User: The created user
        """
        hashed_password = AuthService.get_password_hash(user_data.password)

        user = User(
            email=user_data.email,
            password_hash=hashed_password,
            specialization=user_data.specialization,
            ros_experience_level=user_data.ros_experience_level,
            focus_area=user_data.focus_area,
            language_preference=user_data.language_preference,
            is_active=True,  # New users are active by default
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def create_oauth_user(db: Session, email: str, provider: str) -> User:
        """Create a new user from OAuth authentication.

        Args:
            db: Database session
            email: User's email from OAuth provider
            provider: OAuth provider name (google, github)

        Returns:
            User: The created user
        """
        user = User(
            email=email,
            password_hash=None,  # OAuth users don't have passwords
            is_active=True,  # New OAuth users are active by default
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    @staticmethod
    def create_tokens(user_id: UUID) -> Tuple[str, str, int]:
        """Create access and refresh tokens for a user.

        Args:
            user_id: The user's UUID

        Returns:
            Tuple[str, str, int]: (access_token, refresh_token, expires_in_seconds)
        """
        access_token = AuthService.create_access_token(user_id)
        refresh_token = AuthService.create_refresh_token(user_id)
        expires_in = settings.auth_token_expiry_hours * 3600

        return access_token, refresh_token, expires_in


# Singleton instance
auth_service = AuthService()
