\"\"\"OpenAPI specification for the Physical AI & Humanoid Robotics Textbook API.\"\"\"

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid

# Import all models to include in OpenAPI
from src.schemas.user import UserCreate, UserUpdate, UserRead, Token
from src.schemas.module import ModuleCreate, ModuleUpdate, ModuleRead, ModuleName
from src.schemas.lesson import LessonCreate, LessonUpdate, LessonRead
from src.schemas.user_progress import UserProgressCreate, UserProgressUpdate, UserProgressRead
from src.schemas.chatbot_query import ChatbotQueryCreate, ChatbotQueryRead
from src.schemas.lesson_embedding import LessonEmbeddingCreate, LessonEmbeddingRead
from src.schemas.content_translation import ContentTranslationCreate, ContentTranslationRead

def create_openapi_app() -> FastAPI:
    \"\"\"Creates and configures the FastAPI application with OpenAPI metadata.\"\"\"

    app = FastAPI(
        title="Physical AI & Humanoid Robotics Textbook API",
        description=\"\"\"This API serves the backend for an AI-Native Physical AI & Humanoid Robotics Textbook.

        The textbook features:
        - Structured content across four modules (ROS 2, Digital Twin, AI-Robot Brain, VLA)
        - RAG-powered chatbot for querying textbook content
        - User authentication and profiling
        - Content personalization based on user specialization
        - Multi-language support

        This API provides endpoints for managing textbook content, user data, chatbot interactions,
        and content personalization features.\"\"\",
        version="1.0.0",
        contact={
            "name": "Textbook Development Team",
            "url": "https://github.com/physical-ai-robotics/textbook-api",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {
                "url": "https://textbook-api.example.com",
                "description": "Production server"
            },
            {
                "url": "https://staging.textbook-api.example.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ],
        openapi_tags=[
            {
                "name": "Authentication",
                "description": "Endpoints for user authentication and token management"
            },
            {
                "name": "Users",
                "description": "Operations with users, including profile management"
            },
            {
                "name": "Modules",
                "description": "Operations with textbook modules (ROS 2, Digital Twin, AI-Robot Brain, VLA)"
            },
            {
                "name": "Lessons",
                "description": "Operations with textbook lessons and content"
            },
            {
                "name": "User Progress",
                "description": "Operations for tracking user progress through lessons"
            },
            {
                "name": "Chatbot Queries",
                "description": "Operations for managing chatbot interactions and queries"
            },
            {
                "name": "Content",
                "description": "Operations for managing content, embeddings, and translations"
            }
        ]
    )

    return app

# Example usage models for documentation purposes
class APIResponseExample(BaseModel):
    \"\"\"Example response model for documentation.\"\"\"
    message: str
    data: Optional[Dict[str, Any]] = None

# Define common response examples
USER_EXAMPLE = {
    "user_id": str(uuid.uuid4()),
    "email": "student@example.com",
    "specialization": ["ROS 2", "Computer Vision"],
    "ros_experience_level": "intermediate",
    "focus_area": "both",
    "language_preference": "en"
}

MODULE_EXAMPLE = {
    "module_id": str(uuid.uuid4()),
    "name": "ROS2",
    "description": "Fundamentals of Robot Operating System 2",
    "order_index": 1
}

LESSON_EXAMPLE = {
    "lesson_id": str(uuid.uuid4()),
    "module_id": str(uuid.uuid4()),
    "title": "Introduction to ROS 2 Architecture",
    "learning_objectives": [
        "Understand ROS 2 node communication",
        "Learn about topics, services, and actions"
    ],
    "content_markdown": "# Introduction to ROS 2\\n\\nROS 2 (Robot Operating System 2) ...",
    "code_examples": [
        {
            "language": "python",
            "code": "import rclpy\\nrclpy.init()",
            "description": "Basic ROS 2 node initialization"
        }
    ],
    "diagrams": [
        "https://example.com/ros2-architecture.png"
    ],
    "order_index": 1
}

USER_PROGRESS_EXAMPLE = {
    "progress_id": str(uuid.uuid4()),
    "user_id": str(uuid.uuid4()),
    "lesson_id": str(uuid.uuid4()),
    "completed_at": "2023-10-15T10:30:00Z",
    "bookmarked": True,
    "time_spent_seconds": 1800
}

CHATBOT_QUERY_EXAMPLE = {
    "query_id": str(uuid.uuid4()),
    "user_id": str(uuid.uuid4()),
    "query_text": "Explain ROS 2 node communication",
    "retrieved_passages": [
        {
            "lesson_id": str(uuid.uuid4()),
            "passage_text": "In ROS 2, nodes communicate through topics, services, and actions...",
            "similarity_score": 0.92
        }
    ],
    "response_text": "In ROS 2, nodes communicate through topics, services, and actions...",
    "response_generation_time_ms": 450
}

# These examples can be used in the API documentation to show sample requests and responses
API_EXAMPLES = {
    "user": USER_EXAMPLE,
    "module": MODULE_EXAMPLE,
    "lesson": LESSON_EXAMPLE,
    "user_progress": USER_PROGRESS_EXAMPLE,
    "chatbot_query": CHATBOT_QUERY_EXAMPLE
}