"""Authentication middleware for request processing."""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for authentication and request logging.

    This middleware:
    1. Logs all incoming requests
    2. Handles authentication errors gracefully
    3. Adds request context for debugging
    """

    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request through the middleware.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the next handler or error response
        """
        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else None,
            }
        )

        try:
            # Process request
            response = await call_next(request)
            return response

        except HTTPException as exc:
            # Handle HTTP exceptions
            logger.warning(
                f"HTTP exception: {exc.status_code} - {exc.detail}",
                extra={
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "path": request.url.path,
                }
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

        except Exception as exc:
            # Handle unexpected exceptions
            logger.error(
                f"Unexpected error: {str(exc)}",
                exc_info=True,
                extra={"path": request.url.path}
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests.

    This is a placeholder for future rate limiting implementation.
    Could use libraries like slowapi or redis-based rate limiting.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request through rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response from the next handler or rate limit error
        """
        # TODO: Implement actual rate limiting logic
        # For now, just pass through
        response = await call_next(request)
        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware for fine-grained control.

    Note: FastAPI's built-in CORSMiddleware is already being used in main.py.
    This is a placeholder for any additional CORS customization if needed.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request through CORS handling.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            Response: The response with CORS headers
        """
        response = await call_next(request)
        return response
