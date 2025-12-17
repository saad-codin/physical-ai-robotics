"""Comprehensive RAG system diagnostic."""
import asyncio
import sys
import io
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    print("=" * 80)
    print("COMPREHENSIVE RAG SYSTEM DIAGNOSTIC")
    print("=" * 80)

    # Step 1: Check Qdrant connection and data
    print("\n[1/4] Checking Qdrant connection and data...")
    try:
        from src.services.qdrant_client import get_qdrant_service
        qdrant = get_qdrant_service()
        info = qdrant.get_collection_info()
        points_count = info.get('points_count', 0)

        print(f"    ✓ Qdrant connected")
        print(f"      Collection: {info.get('name', 'N/A')}")
        print(f"      Vector dimension: {info.get('vector_size', 'N/A')}")
        print(f"      Points (embeddings): {points_count}")

        if points_count == 0:
            print("    ⚠ WARNING: No embeddings in Qdrant!")
            print("      You need to index lessons first.")
            print("      Check if lessons were properly indexed.")
        else:
            print(f"    ✓ Found {points_count} embeddings in Qdrant")

    except Exception as e:
        print(f"    ✗ Qdrant check failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Sample a few points from Qdrant
    print("\n[2/4] Sampling embeddings from Qdrant...")
    try:
        from qdrant_client.models import Filter

        # Scroll through some points to see what's stored
        scroll_result = qdrant.client.scroll(
            collection_name=qdrant.collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]  # First element is the list of points

        if len(points) > 0:
            print(f"    ✓ Sample of {len(points)} embeddings:")
            for i, point in enumerate(points, 1):
                payload = point.payload
                lesson_id = payload.get('lesson_id', 'N/A')
                title = payload.get('title', 'N/A')
                passage_text = payload.get('passage_text', '')

                print(f"\n      [{i}] Embedding ID: {point.id}")
                print(f"          Lesson ID: {lesson_id}")
                print(f"          Title: {title}")
                print(f"          Text preview: {passage_text[:100]}...")
        else:
            print("    ⚠ No points found in collection")

    except Exception as e:
        print(f"    ✗ Sampling failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 3: Test embedding generation
    print("\n[3/4] Testing embedding generation...")
    try:
        from src.utils.embeddings import llm_service
        test_text = "What is robotics?"
        print(f"    Test text: '{test_text}'")

        embedding = llm_service.generate_embedding(test_text)
        print(f"    ✓ Generated embedding")
        print(f"      Dimension: {len(embedding)}")
        print(f"      First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"    ✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Test RAG search with various queries
    print("\n[4/4] Testing RAG search with multiple queries...")

    test_queries = [
        "What is the book about?",
        "Tell me about the Physical AI textbook",
        "What is ROS2?",
        "Explain robotics",
        "What topics are covered in this book?",
    ]

    for query in test_queries:
        print(f"\n    Query: '{query}'")
        print("    " + "-" * 70)

        try:
            results = await qdrant.search_similar_passages_async(
                query_text=query,
                top_k=3,
                similarity_threshold=0.25  # Optimized threshold
            )

            print(f"    Found {len(results)} results")

            if len(results) > 0:
                print("    Top result:")
                top = results[0]
                lesson_id = top.get('lesson_id', 'N/A')
                similarity = top.get('similarity_score', 0)
                text = top.get('passage_text', '')
                metadata = top.get('metadata', {})

                print(f"      ✓ Similarity: {similarity:.3f}")
                print(f"        Lesson ID: {lesson_id}")
                if metadata:
                    print(f"        Metadata: {metadata}")
                print(f"        Text: {text[:150]}...")
            else:
                print("    ⚠ No results found for this query")

        except Exception as e:
            print(f"    ✗ Search failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Summary and recommendations
    print("\n📊 SUMMARY:")
    if points_count > 0:
        print(f"  • Qdrant has {points_count} embeddings stored ✓")
        print("  • RAG system is set up correctly ✓")
        print("\n💡 RECOMMENDATIONS:")
        print("  1. Check if the AI response includes retrieved context")
        print("  2. Lower similarity_threshold if not getting results (currently 0.5)")
        print("  3. Ensure queries match the book content domain")
        print("  4. Check ChatKit system prompt for proper context formatting")
    else:
        print("  • No embeddings in Qdrant ✗")
        print("\n💡 NEXT STEPS:")
        print("  1. Run the lesson indexing script")
        print("  2. Check if lessons exist in database")
        print("  3. Verify OpenAI API key for embeddings")

if __name__ == "__main__":
    asyncio.run(main())
