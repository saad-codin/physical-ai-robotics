import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.schemas.chatbot_query import ChatbotQueryRead, ChatbotQueryCreate
from src.schemas.chatbot import ChatbotQueryRequest, ChatbotQueryResponse
from src.crud import chatbot_query as chatbot_query_crud
from src.models.chatbot_query import ChatbotQuery as DBChatbotQuery
from src.models.user import User as DBUser
from src.core.auth import get_current_active_user, get_optional_current_active_user
from src.services.chatbot import chatbot_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=ChatbotQueryRead, status_code=status.HTTP_201_CREATED)
async def create_chatbot_query(
    chatbot_query_in: ChatbotQueryCreate,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    # Ensure the query is for the current user
    if chatbot_query_in.user_id and str(chatbot_query_in.user_id) != str(current_user.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create query for another user"
        )
    chatbot_query_in.user_id = current_user.user_id # Ensure correct user_id

    return await chatbot_query_crud.create_chatbot_query(db, chatbot_query=chatbot_query_in)


@router.post("/chat", response_model=ChatbotQueryResponse)
async def chat_with_bot(
    query_request: ChatbotQueryRequest,
    current_user: Optional[DBUser] = Depends(get_optional_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Endpoint to chat with the RAG-enabled bot.

    This endpoint processes the user query by:
    1. Searching for relevant passages in the vector database (Qdrant)
    2. Generating an LLM response based on the passages
    3. Saving the query and response to the database
    """
    try:
        # Search for relevant passages using the RAG service
        retrieved_passages = await chatbot_service.search_relevant_passages(
            query_text=query_request.query_text,
            top_k=5,  # Retrieve top 5 passages
            similarity_threshold=0.5
        )

        # Generate response using LLM
        llm_response = await chatbot_service.generate_response(
            query_text=query_request.query_text,
            retrieved_passages=retrieved_passages,
            model_name="gpt-3.5-turbo"  # or get from settings
        )

        # Create a record in the database
        chatbot_query_in = ChatbotQueryCreate(
            user_id=current_user.user_id if current_user else None,
            query_text=query_request.query_text,
            retrieved_passages=retrieved_passages,
            response_text=llm_response["response_text"],
            response_generation_time_ms=llm_response["response_generation_time_ms"],
        )
        
        created_query = await chatbot_query_crud.create_chatbot_query(
            db, chatbot_query=chatbot_query_in
        )

        # Return formatted response
        return ChatbotQueryResponse(
            query_id=created_query.query_id,
            query_text=created_query.query_text,
            response_text=llm_response["response_text"],
            retrieved_passages=[
                {
                    "lesson_id": p["lesson_id"],
                    "passage_text": p["passage_text"],
                    "similarity_score": p["similarity_score"]
                }
                for p in retrieved_passages
            ],
            response_generation_time_ms=llm_response["response_generation_time_ms"],
            created_at=created_query.created_at
        )

    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat query"
        )

@router.get("/me", response_model=List[ChatbotQueryRead])
async def get_my_chatbot_queries(
    skip: int = 0,
    limit: int = 100,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    queries = await chatbot_query_crud.get_chatbot_queries(
        db, user_id=current_user.user_id, skip=skip, limit=limit
    )
    return queries

@router.get("/{query_id}", response_model=ChatbotQueryRead)
async def get_chatbot_query_by_id(
    query_id: str,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    query = await chatbot_query_crud.get_chatbot_query(db, query_id=query_id)
    if not query:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbot query not found")
    if str(query.user_id) != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this query")
    return query

@router.delete("/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chatbot_query(
    query_id: str,
    current_user: DBUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    query = await chatbot_query_crud.get_chatbot_query(db, query_id=query_id)
    if not query:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chatbot query not found")
    if str(query.user_id) != str(current_user.user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this query")

    await chatbot_query_crud.delete_chatbot_query(db, query_id=query_id)
    return None
