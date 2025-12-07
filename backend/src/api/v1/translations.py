"""Simple Urdu translation API using OpenAI GPT-3.5-turbo."""
import logging
from typing import Dict, Any
import openai
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from uuid import UUID

from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize OpenAI client
client = openai.OpenAI(api_key=settings.openai_api_key)

class UrduTranslationRequest(BaseModel):
    """Request model for Urdu translation."""
    text: str

class UrduTranslationResponse(BaseModel):
    """Response model for Urdu translation."""
    original_text: str
    translated_text: str
    language: str = "ur"

@router.post("/translate-to-urdu", response_model=UrduTranslationResponse)
async def translate_to_urdu(request: UrduTranslationRequest):
    """
    Translate text to Urdu using OpenAI GPT-3.5-turbo.

    This endpoint takes any input text and translates it to Urdu.
    """
    try:
        # Create a translation prompt for GPT-3.5-turbo
        prompt = f"Translate the following text to Urdu (Urdu script):\n\n{request.text}\n\nTranslation:"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given text accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000,  # Adjust based on expected output length
            temperature=0.3,  # Lower temperature for more consistent translations
        )

        translated_text = response.choices[0].message.content.strip()

        return UrduTranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            language="ur"
        )

    except Exception as e:
        logger.error(f"Error in Urdu translation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating to Urdu: {str(e)}"
        )

# Additional endpoint for translating content specifically for lessons
class LessonTranslationRequest(BaseModel):
    """Request model for lesson content translation."""
    title: str
    content: str

class LessonTranslationResponse(BaseModel):
    """Response model for lesson content translation."""
    original_title: str
    original_content: str
    translated_title: str
    translated_content: str
    language: str = "ur"

@router.post("/translate-lesson", response_model=LessonTranslationResponse)
async def translate_lesson_content(request: LessonTranslationRequest):
    """
    Translate lesson content (title and content) to Urdu using OpenAI GPT-3.5-turbo.
    """
    try:
        # Translate title
        title_prompt = f"Translate the following lesson title to Urdu (Urdu script):\n\n{request.title}\n\nTranslation:"

        title_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given lesson title accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar."
                },
                {
                    "role": "user",
                    "content": title_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )

        translated_title = title_response.choices[0].message.content.strip()

        # Translate content
        content_prompt = f"Translate the following lesson content to Urdu (Urdu script):\n\n{request.content}\n\nTranslation:"

        content_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given lesson content accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar. Maintain any technical terminology appropriately."
                },
                {
                    "role": "user",
                    "content": content_prompt
                }
            ],
            max_tokens=1500,
            temperature=0.3,
        )

        translated_content = content_response.choices[0].message.content.strip()

        return LessonTranslationResponse(
            original_title=request.title,
            original_content=request.content,
            translated_title=translated_title,
            translated_content=translated_content,
            language="ur"
        )

    except Exception as e:
        logger.error(f"Error in lesson translation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating lesson to Urdu: {str(e)}"
        )