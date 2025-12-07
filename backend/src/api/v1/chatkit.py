"""ChatKit integration for OpenAI ChatKit framework with RAG capabilities."""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.core.auth import get_current_active_user, get_optional_current_active_user
from src.models.user import User as DBUser
from src.services.qdrant_client import get_qdrant_service
from src.services.chatbot import chatbot_service
from src.core.config import settings
import openai

logger = logging.getLogger(__name__)

router = APIRouter()

security = HTTPBearer()


@router.post("/session")
async def create_chatkit_session(
    current_user: Optional[DBUser] = Depends(get_optional_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a ChatKit session for the current user.

    This endpoint returns a client secret that can be used by the ChatKit frontend
    to establish a connection with the backend.
    """
    try:
        # Create a proper client secret for ChatKit
        import uuid
        client_secret = f"sk-chatkit-{uuid.uuid4()}-secret"

        return {
            "client_secret": client_secret,
            "user_id": str(current_user.user_id) if current_user else "anonymous",
            "session_id": f"session_{uuid.uuid4()}",
            "expires_at": (datetime.now()).isoformat(),
            "scope": "chatkit:user"
        }
    except Exception as e:
        logger.error(f"Error creating ChatKit session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating ChatKit session"
        )


@router.post("/chat")
async def chatkit_chat(
    request: Request,
    current_user: Optional[DBUser] = Depends(get_optional_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Chat endpoint for ChatKit that processes user messages with RAG capabilities.
    """
    try:
        # Get the request body
        body = await request.json()

        # Extract the query text from the request
        query_text = body.get("message", "") or body.get("query", "") or body.get("text", "")

        if not query_text.strip():
            return {
                "type": "thread.stream_event",
                "event": {
                    "type": "text",
                    "text": "Hello! I'm your AI Robotics Tutor. Ask me anything about robotics concepts, programming, or theory!"
                }
            }

        # Initialize OpenAI client
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        # First, try to retrieve relevant passages from Qdrant
        qdrant_service = get_qdrant_service()

        try:
            # Search for relevant passages in the knowledge base
            retrieved_passages = await qdrant_service.search_similar_passages(
                query_text=query_text,
                top_k=5,
                similarity_threshold=0.5
            )

            if retrieved_passages:
                # Build context from retrieved passages
                context = "\\n\\n".join([passage.get("payload", {}).get("text", "") for passage in retrieved_passages])

                # Create a RAG-enhanced prompt
                system_message = f"""You are an AI Robotics Tutor. Use the following context to answer the user's question about robotics concepts. If the answer is not in the context, use your general knowledge but mention that it's not from the provided materials.

Context:
{context}

Be helpful, educational, and accurate in your responses about robotics, AI, and related topics."""
            else:
                # No relevant passages found, use general robotics knowledge
                system_message = "You are an AI Robotics Tutor. Answer the user's question about robotics concepts, programming, or theory to the best of your knowledge. Be helpful, educational, and accurate."

        except Exception as qdrant_error:
            logger.warning(f"Qdrant search failed: {qdrant_error}, proceeding with general knowledge")
            # If Qdrant is unavailable, proceed with general knowledge
            system_message = "You are an AI Robotics Tutor. Answer the user's question about robotics concepts, programming, or theory to the best of your knowledge. Be helpful, educational, and accurate."

        # Call OpenAI API with the prepared context
        response = await client.chat.completions.create(
            model=settings.openai_model or "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query_text}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        ai_response = response.choices[0].message.content

        return {
            "type": "thread.stream_event",
            "event": {
                "type": "text",
                "text": ai_response
            }
        }

    except Exception as e:
        logger.error(f"Error in ChatKit chat: {e}", exc_info=True)
        return {
            "type": "thread.stream_event",
            "event": {
                "type": "text",
                "text": f"Sorry, I encountered an error processing your request: {str(e)}"
            }
        }


@router.post("/translate-to-urdu")
async def translate_to_urdu(
    request: Request,
    current_user: Optional[DBUser] = Depends(get_optional_current_active_user)
):
    """
    Translation endpoint that translates text to Urdu using OpenAI.
    """
    try:
        # Get the request body
        body = await request.json()

        # Extract the text to translate
        text_to_translate = body.get("text", "")

        if not text_to_translate.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text to translate is required"
            )

        # Initialize OpenAI client
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        # Create a translation prompt
        system_message = "You are a professional translator. Translate the given text accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar."
        user_message = f"Translate the following text to Urdu (Urdu script):\n\n{text_to_translate}\n\nTranslation:"

        # Call OpenAI API for translation
        response = await client.chat.completions.create(
            model=settings.openai_model or "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent translations
        )

        translated_text = response.choices[0].message.content.strip()

        return {
            "original_text": text_to_translate,
            "translated_text": translated_text,
            "target_language": "ur",
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in Urdu translation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating to Urdu: {str(e)}"
        )


@router.post("/translate-lesson-content")
async def translate_lesson_content(
    request: Request,
    current_user: Optional[DBUser] = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Translate lesson content (title and body) to Urdu using OpenAI.
    """
    try:
        # Get the request body
        body = await request.json()

        # Extract the content to translate
        title = body.get("title", "")
        content = body.get("content", "")

        if not title.strip() and not content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either title or content is required for translation"
            )

        # Initialize OpenAI client
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        translations = {}

        # Translate title if provided
        if title.strip():
            title_response = await client.chat.completions.create(
                model=settings.openai_model or "gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the given lesson title accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar."
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following lesson title to Urdu (Urdu script):\n\n{title}\n\nTranslation:"
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            translations["title"] = title_response.choices[0].message.content.strip()

        # Translate content if provided
        if content.strip():
            content_response = await client.chat.completions.create(
                model=settings.openai_model or "gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the given lesson content accurately to Urdu while preserving the meaning and context. Use proper Urdu script and grammar. Maintain any technical terminology appropriately."
                    },
                    {
                        "role": "user",
                        "content": f"Translate the following lesson content to Urdu (Urdu script):\n\n{content}\n\nTranslation:"
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            translations["content"] = content_response.choices[0].message.content.strip()

        return {
            "original_title": title,
            "original_content": content,
            "translated_title": translations.get("title", ""),
            "translated_content": translations.get("content", ""),
            "target_language": "ur",
            "success": True
        }

    except Exception as e:
        logger.error(f"Error in lesson content translation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating lesson content to Urdu: {str(e)}"
        )