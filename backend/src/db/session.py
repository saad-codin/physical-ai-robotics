"""Database session configuration for SQLAlchemy."""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/textbook_db")

# Convert to async URL if needed (replace postgresql:// with postgresql+asyncpg://)
# Also handle query parameters that asyncpg doesn't support
if DATABASE_URL.startswith("postgresql://"):
    # Split the URL to handle query parameters
    if '?' in DATABASE_URL:
        base_url, query_params = DATABASE_URL.split('?', 1)
        # Remove query parameters that asyncpg doesn't support
        filtered_params = []
        for param in query_params.split('&'):
            if not param.startswith(('sslmode=', 'channel_binding=')):
                filtered_params.append(param)
        query_part = '&'.join(filtered_params)
        DATABASE_URL = base_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if query_part:
            DATABASE_URL += '?' + query_part
    else:
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async SQLAlchemy engine
# pool_pre_ping: verify connections before using them from the pool
# pool_size: number of connections to maintain in the pool
# max_overflow: number of connections that can be created beyond pool_size
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,  # Set to True for SQL query logging during development
)

# Create async sessionmaker
# autocommit=False: require explicit commit() calls
# autoflush=False: disable automatic flush before queries
# bind: bind to the engine
async_session = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function to get async database session.

    Yields a database session and ensures it's closed after use.
    Used with FastAPI's Depends() for dependency injection.

    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with async_session() as db:
        try:
            yield db
        finally:
            await db.close()
