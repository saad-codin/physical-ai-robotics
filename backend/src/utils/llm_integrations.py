"""Utilities for integrating with Large Language Models (LLMs) for RAG and response generation."""

import os
from typing import List, Dict, Any, Optional

# Constants for LLM and Embedding models
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize_llm_client(model_name: str = DEFAULT_LLM_MODEL):
    """Initializes and returns an LLM client based on the model name."""
    if "claude" in model_name.lower():
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError("Please install the 'anthropic' library to use Claude models.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    elif "gpt" in model_name.lower() or "text-embedding" in model_name.lower():
        try:
            from openai import OpenAI
            return OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            raise ImportError("Please install the 'openai' library to use OpenAI models.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    else:
        raise ValueError(f"Unsupported LLM model: {model_name}")

# Add these types to the global scope to avoid NameError
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Define as None if import fails

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # Define as None if import fails

def generate_embeddings(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """Generates embeddings for a list of texts using the specified model."""
    client = initialize_llm_client(model_name)
    if isinstance(client, OpenAI):
        response = client.embeddings.create(input=texts, model=model_name)
        return [data.embedding for data in response.data]
    else:
        raise ValueError(f"Embedding generation not supported for client type: {type(client)}")

def generate_llm_response(
    query_text: str,
    retrieved_passages: List[Dict[str, Any]],
    model_name: str = DEFAULT_LLM_MODEL,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generates an LLM response based on the query and retrieved passages.
    Extracts citations from the response.
    """
    client = initialize_llm_client(model_name)

    context = "\n\n".join([p["passage_text"] for p in retrieved_passages])

    if retrieved_passages:
        # If we have retrieved passages, use RAG approach
        prompt_template = f"""
        You are an expert in Physical AI and Humanoid Robotics.
        Answer the user's question based *only* on the provided context.
        If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided information."
        Cite your sources using Markdown footnotes, linking to the relevant passage. For example: [1](ref:lesson_id_1).
        Ensure the citations directly correspond to the provided retrieved passages.

        Context:
        {context}

        User Question: {query_text}

        Answer:
        """
    else:
        # If no passages retrieved, ask the LLM directly without RAG constraints
        prompt_template = f"""
        You are an expert in Physical AI and Humanoid Robotics.
        Please answer the user's question to the best of your knowledge.

        User Question: {query_text}

        Answer:
        """

    response_text = ""
    if isinstance(client, OpenAI):
        system_message = "You are an expert in Physical AI and Humanoid Robotics." if not retrieved_passages else "You are an expert in Physical AI and Humanoid Robotics."
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_template}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = response.choices[0].message.content
    elif isinstance(client, Anthropic):
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt_template}
            ]
        )
        response_text = response.content[0].text
    else:
        raise ValueError(f"Response generation not supported for client type: {type(client)}")

    # Basic citation extraction (only if there were retrieved passages)
    citations = []
    if retrieved_passages:
        import re
        citation_pattern = re.compile(r"\[(\d+)\]\(ref:(lesson_id_[0-9a-f-]+)\)")

        # Replace lesson_id_X with actual lesson_id from retrieved_passages
        for i, passage in enumerate(retrieved_passages):
            placeholder = f"lesson_id_{i+1}"
            actual_id = str(passage.get("lesson_id", f"unknown_lesson_{i+1}")) # Ensure lesson_id is a string
            response_text = response_text.replace(placeholder, actual_id)

        # Re-run pattern matching after replacement
        for match in citation_pattern.finditer(response_text):
            citation_number, lesson_id = match.groups()
            citations.append({"citation_number": int(citation_number), "lesson_id": lesson_id})

    return {
        "response_text": response_text,
        "citations": citations
    }