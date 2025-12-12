"""Simple script to verify RAG system is working."""
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
    print("=" * 60)
    print("RAG VERIFICATION TEST")
    print("=" * 60)

    # Step 1: Check database connection
    print("\n[1/5] Checking database connection...")
    try:
        from src.db.session import async_session
        async with async_session() as db:
            print("    ✓ Database connection successful")
    except Exception as e:
        print(f"    ✗ Database connection failed: {e}")
        return

    # Step 2: Check Qdrant connection
    print("\n[2/5] Checking Qdrant connection...")
    try:
        from src.services.qdrant_client import get_qdrant_service
        qdrant = get_qdrant_service()
        info = qdrant.get_collection_info()
        print(f"    ✓ Qdrant connected")
        print(f"      Collection: {info.get('name', 'N/A')}")
        print(f"      Points: {info.get('points_count', 0)}")
    except Exception as e:
        print(f"    ✗ Qdrant connection failed: {e}")
        return

    # Step 3: Check lessons in database
    print("\n[3/5] Checking lessons in database...")
    try:
        from src.db.session import async_session
        from sqlalchemy import select, func
        from src.models.lesson import Lesson

        async with async_session() as db:
            result = await db.execute(select(func.count(Lesson.lesson_id)))
            lesson_count = result.scalar()
            print(f"    Lessons in database: {lesson_count}")

            if lesson_count == 0:
                print("    ⚠ No lessons found. You need to import docs first!")
                print("      Run: python import_docs.py")
            else:
                print(f"    ✓ Found {lesson_count} lessons")
    except Exception as e:
        print(f"    ✗ Failed to check lessons: {e}")
        return

    # Step 4: Test embedding generation
    print("\n[4/5] Testing embedding generation...")
    try:
        from src.utils.embeddings import llm_service
        test_text = "What is ROS2?"
        embedding = llm_service.generate_embedding(test_text)
        print(f"    ✓ Generated embedding (dimension: {len(embedding)})")
    except Exception as e:
        print(f"    ✗ Embedding generation failed: {e}")
        return

    # Step 5: Test RAG search
    print("\n[5/5] Testing RAG search...")
    try:
        from src.services.qdrant_client import get_qdrant_service
        qdrant = get_qdrant_service()

        query_text = "What is ROS2?"
        results = await qdrant.search_similar_passages_async(
            query_text=query_text,
            top_k=3,
            similarity_threshold=0.3
        )

        print(f"    Query: '{query_text}'")
        print(f"    Results found: {len(results)}")

        if len(results) > 0:
            print("    ✓ RAG search working!")
            print("\n    Top result:")
            top = results[0]
            print(f"      Lesson ID: {top.get('lesson_id', 'N/A')}")
            print(f"      Similarity: {top.get('similarity_score', 0):.3f}")
            print(f"      Text preview: {top.get('passage_text', '')[:100]}...")
        else:
            print("    ⚠ No results found (may need to index docs)")
    except Exception as e:
        print(f"    ✗ RAG search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
