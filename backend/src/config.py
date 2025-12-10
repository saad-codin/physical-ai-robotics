"""Environment configuration using Pydantic settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Any
from pydantic import model_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Constitution Principle: All configuration must be externalized via .env
    for reproducibility across development/staging/production environments.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database Configuration
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "lesson_embeddings"
    qdrant_vector_dimension: int = 1536  # text-embedding-3-large dimension

    # LLM Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000

    # Authentication Configuration
    auth_secret: str
    auth_token_expiry_hours: int = 24
    auth_refresh_token_expiry_days: int = 30

    # CORS Configuration
    frontend_url: str = "http://localhost:3000"
    allowed_origins: List[str] = []

    @model_validator(mode='before')
    @classmethod
    def set_allowed_origins(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get('allowed_origins'):
            frontend_url = values.get('frontend_url', 'http://localhost:3000')
            # Always include both production and localhost
            values['allowed_origins'] = [
                frontend_url,
                'http://localhost:3000',
                'https://physical-ai-robotics-six.vercel.app'
            ]
        return values

    # Application Configuration
    environment: str = "development"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"

    # RAG Configuration
    rag_top_k_results: int = 5
    rag_similarity_threshold: float = 0.7

    # Multi-language Support
    default_language: str = "en"
    supported_languages: list[str] = ["en", "zh", "es", "fr", "de", "ja", "ko", "ur"]


# Singleton instance of settings
settings = Settings()
