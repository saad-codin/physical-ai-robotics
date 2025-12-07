"""Content translation schemas for multi-language support."""
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional


class ContentTranslationBase(BaseModel):
    """Base content translation schema."""
    lesson_id: UUID
    language_code: str = Field(..., pattern=r"^[a-z]{2}(-[A-Z]{2})?$", max_length=10)  # e.g., en, en-US, zh, zh-CN
    translated_title: str = Field(..., max_length=500)
    translated_content_markdown: str


class ContentTranslationCreate(ContentTranslationBase):
    """Schema for creating a content translation."""
    pass


class ContentTranslationUpdate(BaseModel):
    """Schema for updating a content translation."""
    translated_title: Optional[str] = Field(None, max_length=500)
    translated_content_markdown: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class ContentTranslationResponse(ContentTranslationBase):
    """Schema for content translation response."""
    translation_id: UUID
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TranslationRequest(BaseModel):
    """Schema for requesting translation of content."""
    lesson_id: UUID
    target_language: str = Field(..., pattern=r"^[a-z]{2}(-[A-Z]{2})?$", max_length=10)
    use_existing_translation: bool = True  # Whether to use existing translation if available


class TranslationResponse(BaseModel):
    """Schema for translation response."""
    lesson_id: UUID
    language_code: str
    translated_title: str
    translated_content_markdown: str
    is_new_translation: bool
    created_at: datetime


class SupportedLanguagesResponse(BaseModel):
    """Schema for supported languages response."""
    supported_languages: list[dict[str, str]]  # List of {code: name} pairs
    default_language: str