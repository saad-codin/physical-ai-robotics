"""Test RAG search functionality."""
import asyncio
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.services.qdrant_client import get_qdrant_service
from src.utils.embeddings import llm_service

async def test_search():
    """Test RAG search."""
    print("=" * 60)
    print("TESTING RAG SEARCH")
    print("=" * 60)

    qdrant = get_qdrant_service()

    # Test queries
    queries = [
        "What is ROS2?",
        "Explain digital twin",
        "What are vision-language-action models?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        try:
            # Use the async search method with optimized threshold
            results = await qdrant.search_similar_passages_async(
                query_text=query,
                top_k=3,
                similarity_threshold=0.25  # Optimized threshold (94.1% coverage)
            )

            print(f"Found {len(results)} results:\n")

            for i, result in enumerate(results, 1):
                lesson_id = result.get('lesson_id', 'N/A')
                similarity = result.get('similarity_score', 0)
                text = result.get('passage_text', '')

                print(f"[{i}] Similarity: {similarity:.3f}")
                print(f"    Lesson ID: {lesson_id}")
                print(f"    Text: {text[:150]}...")
                print()

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_search())
