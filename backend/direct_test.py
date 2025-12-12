"""Direct test of Qdrant search without async complications."""
import sys
import io
from qdrant_client import QdrantClient
from src.config import settings
from src.utils.embeddings import llm_service

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test():
    print("Connecting to Qdrant...")

    # Create client
    import re
    host_match = re.search(r'https://([^/:]+)', settings.qdrant_url)
    if host_match:
        host = host_match.group(1)
        client = QdrantClient(
            host=host,
            api_key=settings.qdrant_api_key,
            https=True,
            timeout=30.0,
        )
    else:
        print("Could not parse Qdrant URL")
        return

    print(f"Connected to: {host}")

    # Check collection
    info = client.get_collection(settings.qdrant_collection_name)
    print(f"Collection: {settings.qdrant_collection_name}")
    print(f"Points: {info.points_count}")

    # Generate embedding for test query
    print("\nGenerating embedding for query: 'What is ROS2?'")
    query_vector = llm_service.generate_embedding("What is ROS2?")
    print(f"Embedding dimension: {len(query_vector)}")

    # Search
    print("\nSearching...")
    results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=3,
        score_threshold=0.3,
    )

    print(f"Results found: {len(results.points)}\n")

    for i, point in enumerate(results.points, 1):
        print(f"[{i}] Score: {point.score:.3f}")
        print(f"    Lesson ID: {point.payload.get('lesson_id')}")
        print(f"    Text: {point.payload.get('passage_text', '')[:150]}...")
        print()

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
