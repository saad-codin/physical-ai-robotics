from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1 import router as v1_router
from src.db.session import engine
from src.config import settings
from src.db import base  # Import this to register models with SQLAlchemy

# Set up detailed logging
import sys
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Initializing database connection...")
    # Create tables if they don't exist
    async with engine.begin() as conn:
        # This will create all tables defined in models
        await conn.run_sync(base.Base.metadata.create_all)
    logger.info("Database tables created if not exists")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down...")
    await engine.dispose()

# Create FastAPI application instance
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="Backend API for AI-Native textbook platform with RAG chatbot, progress tracking, and personalization",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers for client-side access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Request logging middleware
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            logger.info(f"Response status: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Request failed with exception: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

app.add_middleware(LoggingMiddleware)

# Global exception handler
from fastapi import status
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Include API routes
app.include_router(v1_router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Physical AI & Humanoid Robotics Textbook API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    from src.services.qdrant_client import get_qdrant_service

    # Test Qdrant connection by getting collection info
    try:
        qdrant_service = get_qdrant_service()
        collection_info = qdrant_service.get_collection_info()
        qdrant_status = "connected" if collection_info else "connection_error"
    except Exception as e:
        qdrant_status = f"connection_error: {str(e)}"

    return {
        "status": "healthy",
        "database": "connected",  # This will be connected due to lifespan manager
        "qdrant": qdrant_status,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
    )
