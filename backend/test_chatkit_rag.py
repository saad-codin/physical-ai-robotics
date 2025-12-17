"""Test ChatKit RAG functionality with various queries."""
import asyncio
import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_rag_queries():
    """Test RAG with various query types."""
    from src.services.qdrant_client import get_qdrant_service

    print("=" * 80)
    print("TESTING CHATKIT RAG WITH VARIOUS QUERIES")
    print("=" * 80)

    qdrant = get_qdrant_service()

    # Test different types of queries
    test_cases = [
        {
            "query": "What is the book about?",
            "description": "Generic book query",
            "expected": "Should return general textbook overview"
        },
        {
            "query": "Tell me about this textbook",
            "description": "Book context query",
            "expected": "Should understand textbook context"
        },
        {
            "query": "What topics are covered?",
            "description": "Table of contents query",
            "expected": "Should list main topics"
        },
        {
            "query": "What is ROS2?",
            "description": "Specific technical query",
            "expected": "Should return ROS2 content"
        },
        {
            "query": "Explain AI-Robot Brain",
            "description": "Core concept query",
            "expected": "Should return brain architecture info"
        },
        {
            "query": "What are VLA models?",
            "description": "Advanced topic query",
            "expected": "Should return VLA information"
        },
    ]

    book_title = "Physical AI & Humanoid Robotics Textbook"
    similarity_threshold = 0.35  # Lowered threshold

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected = test_case["expected"]

        print(f"\n[{i}/{len(test_cases)}] {description}")
        print(f"Query: '{query}'")
        print(f"Expected: {expected}")
        print("-" * 80)

        try:
            # Search with lowered threshold
            results = await qdrant.search_similar_passages_async(
                query_text=query,
                top_k=5,
                similarity_threshold=similarity_threshold
            )

            if results:
                print(f"✓ Found {len(results)} results\n")

                # Show top 2 results
                for j, result in enumerate(results[:2], 1):
                    passage_text = result.get("passage_text", "")
                    similarity = result.get("similarity_score", 0)
                    metadata = result.get("metadata", {})
                    lesson_title = metadata.get("title", "Unknown")

                    print(f"  Result {j}:")
                    print(f"    Lesson: '{lesson_title}'")
                    print(f"    Similarity: {similarity:.3f}")
                    print(f"    Preview: {passage_text[:120]}...")
                    print()

                # Simulate what the AI would see
                print("  📋 Context that AI would receive:")
                context_preview = []
                for result in results[:3]:
                    metadata = result.get("metadata", {})
                    lesson_title = metadata.get("title", "Unknown")
                    similarity = result.get("similarity_score", 0)
                    context_preview.append(
                        f"[From '{lesson_title}' lesson] (Relevance: {similarity:.2f})"
                    )

                print(f"    Book: {book_title}")
                print(f"    Lessons referenced: {', '.join(context_preview)}")
            else:
                print(f"⚠ No results found (threshold: {similarity_threshold})")
                print("  AI will use fallback with book description")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    # Summary
    print("\n📊 KEY IMPROVEMENTS:")
    print("  1. Lowered similarity threshold from 0.5 to 0.35")
    print("  2. Added book title and description to all system prompts")
    print("  3. AI now understands 'book' refers to Physical AI textbook")
    print("  4. Lesson titles are included in context for better citations")
    print("  5. Fallback messages include book overview even without matches")

    print("\n💡 HOW IT WORKS:")
    print("  • User asks: 'What is the book about?'")
    print("  • System searches Qdrant with lower threshold (0.35)")
    print("  • Retrieved passages are shown with lesson names")
    print("  • AI receives book context + retrieved content")
    print("  • AI can answer generic 'book' questions intelligently")

if __name__ == "__main__":
    asyncio.run(test_rag_queries())
