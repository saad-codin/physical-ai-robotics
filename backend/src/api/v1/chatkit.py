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
from src.config import settings
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

        # Get book-level context info
        book_title = "Physical AI & Humanoid Robotics Textbook"
        book_description = "This textbook covers AI-native learning for robotics and AI, including topics like AI-Robot Brain architecture, perception systems, planning, VLA models, digital twins, ROS2, and humanoid robotics."

        try:
            # Search for relevant passages in the knowledge base
            # Lower threshold to 0.35 to catch more results for general queries
            retrieved_passages = await qdrant_service.search_similar_passages_async(
                query_text=query_text,
                top_k=5,
                similarity_threshold=0.35
            )

            if retrieved_passages:
                # Build context from retrieved passages
                # Access passage_text directly from the returned dict
                context_parts = []
                lesson_titles = set()

                for passage in retrieved_passages:
                    passage_text = passage.get("passage_text", "")
                    lesson_id = passage.get("lesson_id", "")
                    similarity_score = passage.get("similarity_score", 0.0)
                    metadata = passage.get("metadata", {})
                    lesson_title = metadata.get("title", "Unknown")

                    lesson_titles.add(lesson_title)

                    context_parts.append(
                        f"[From '{lesson_title}' lesson] (Relevance: {similarity_score:.2f})\n{passage_text}"
                    )

                context = "\n\n---\n\n".join(context_parts)

                # Create a RAG-enhanced prompt with book context
                system_message = f"""You are an AI Robotics Tutor helping students learn from the "{book_title}".

**About this textbook:** {book_description}

When users ask about "the book", "this textbook", or general questions about topics, they're referring to this Physical AI & Humanoid Robotics textbook.

**Relevant content from the textbook:**

{context}

**Instructions:**
- Use the provided textbook content above to answer the user's question
- If asked about "the book" or "what topics are covered", describe the content based on the passages shown
- If the answer isn't fully in the context, supplement with your knowledge but indicate what's from the textbook vs. general knowledge
- Be helpful, educational, and accurate in your responses about robotics, AI, and related topics
- Reference specific lessons when helpful (e.g., "In the Planning lesson...")"""
            else:
                # No relevant passages found, but still provide book context
                system_message = f"""You are an AI Robotics Tutor helping students learn from the "{book_title}".

**About this textbook:** {book_description}

I couldn't find specific passages from the textbook for your query, but I can answer based on general robotics and AI knowledge.

Note: The textbook covers topics including:
- AI-Robot Brain architecture and cognitive systems
- Perception systems (vision, sensors, sensor fusion)
- Planning and decision-making systems
- Vision-Language-Action (VLA) models
- Digital twins and simulation
- ROS2 (Robot Operating System 2)
- Humanoid robotics
- Safety and validation

How can I help you with your question?"""

        except Exception as qdrant_error:
            logger.warning(f"Qdrant search failed: {qdrant_error}, proceeding with general knowledge")
            # If Qdrant is unavailable, still provide book context
            system_message = f"""You are an AI Robotics Tutor helping students learn from the "{book_title}".

**About this textbook:** {book_description}

Note: I'm currently unable to access the textbook content database, but I can answer your robotics and AI questions using general knowledge. The textbook typically covers AI-Robot Brain, perception, planning, VLA models, digital twins, ROS2, and humanoid robotics.

How can I help you?"""

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