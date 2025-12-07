"""LLM integration utilities for embeddings and text generation."""
from openai import OpenAI
from typing import List, Dict, Any
import logging
import time

from src.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations (embeddings and text generation).

    Constitution Principle: RAG chatbot must use semantic embeddings
    and LLM-generated responses citing official documentation.
    """

    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.openai_embedding_model
        self.chat_model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a text passage.

        Args:
            text: The text to embed

        Returns:
            List[float]: The embedding vector (1536 dimensions for text-embedding-3-large)
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )

            embedding = response.data[0].embedding
            logger.info(f"Generated embedding for text of length {len(text)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}")
            raise

    def generate_chatbot_response(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> str:
        """Generate chatbot response using RAG pattern.

        Args:
            query: User's query
            retrieved_passages: List of retrieved passages from Qdrant

        Returns:
            str: Generated response
        """
        try:
            # Build context from retrieved passages
            context = self._build_context(retrieved_passages)

            # Build system prompt with constitution principles
            system_prompt = """You are an AI tutor for Physical AI and Humanoid Robotics.

Your responses must follow these principles:
1. Rigor & Accuracy: All technical claims must cite official documentation
2. Academic Clarity: Use formal terminology and progressive learning approach
3. Reproducibility: Provide copy-paste ready code examples with version pinning

When answering questions:
- Base your answer on the provided context passages
- Cite specific lessons or documentation
- Provide code examples when relevant
- Be concise but thorough
- If the context doesn't contain enough information, acknowledge the limitation

Context passages:
{context}"""

            messages = [
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": query},
            ]

            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            generation_time_ms = int((time.time() - start_time) * 1000)

            response_text = response.choices[0].message.content
            logger.info(
                f"Generated chatbot response in {generation_time_ms}ms "
                f"(tokens: {response.usage.total_tokens})"
            )

            return response_text

        except Exception as e:
            logger.error(f"Error generating chatbot response: {e}")
            raise

    def _build_context(self, retrieved_passages: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved passages.

        Args:
            retrieved_passages: List of retrieved passages with metadata

        Returns:
            str: Formatted context string
        """
        context_parts = []

        for i, passage in enumerate(retrieved_passages, 1):
            lesson_id = passage.get("lesson_id", "unknown")
            passage_text = passage.get("passage_text", "")
            similarity_score = passage.get("similarity_score", 0.0)

            context_parts.append(
                f"[Passage {i}] (Lesson ID: {lesson_id}, Similarity: {similarity_score:.2f})\n"
                f"{passage_text}\n"
            )

        return "\n---\n".join(context_parts)

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> List[str]:
        """Chunk text into smaller passages for embedding.

        Args:
            text: The text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks

        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for delimiter in [". ", ".\n", "! ", "?\n"]:
                    last_delimiter = text.rfind(delimiter, start, end)
                    if last_delimiter != -1:
                        end = last_delimiter + len(delimiter)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        logger.info(f"Chunked text into {len(chunks)} passages")
        return chunks


# Singleton instance
llm_service = LLMService()
