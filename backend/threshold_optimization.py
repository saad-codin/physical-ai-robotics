"""Similarity threshold optimization analysis for RAG system."""
import asyncio
import sys
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.qdrant_client import get_qdrant_service


# Test queries categorized by type
TEST_QUERIES = {
    "General Book Questions": [
        "What is the book about?",
        "What topics are covered in this book?",
        "Tell me about the Physical AI textbook",
        "What can I learn from this textbook?",
    ],
    "Specific Technical Topics": [
        "What is ROS2?",
        "Explain digital twin technology",
        "What are vision-language-action models?",
        "How does sensor fusion work?",
        "What is the AI-Robot Brain?",
    ],
    "Conceptual Understanding": [
        "Explain robotics",
        "What is perception in robotics?",
        "How do robots plan their actions?",
        "What is humanoid robotics?",
    ],
    "Implementation Details": [
        "How to implement ROS2?",
        "Show me code examples",
        "What are the practical applications?",
        "How to build a robot?",
    ]
}

# Thresholds to test
THRESHOLDS_TO_TEST = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


async def test_threshold_performance(
    qdrant_service,
    threshold: float,
    top_k: int = 5
) -> Dict[str, Any]:
    """Test performance metrics for a given threshold."""

    results_by_category = defaultdict(list)
    total_queries = 0
    total_results_found = 0
    total_high_relevance = 0  # Count results with score > 0.6
    similarity_scores = []

    for category, queries in TEST_QUERIES.items():
        for query in queries:
            total_queries += 1
            try:
                results = await qdrant_service.search_similar_passages_async(
                    query_text=query,
                    top_k=top_k,
                    similarity_threshold=threshold
                )

                result_count = len(results)
                total_results_found += result_count

                # Analyze result quality
                if results:
                    top_score = results[0].get('similarity_score', 0)
                    avg_score = sum(r.get('similarity_score', 0) for r in results) / len(results)

                    similarity_scores.extend([r.get('similarity_score', 0) for r in results])

                    # Count high relevance results
                    high_relevance = sum(1 for r in results if r.get('similarity_score', 0) > 0.6)
                    total_high_relevance += high_relevance

                    results_by_category[category].append({
                        'query': query,
                        'count': result_count,
                        'top_score': top_score,
                        'avg_score': avg_score,
                        'high_relevance': high_relevance
                    })
                else:
                    results_by_category[category].append({
                        'query': query,
                        'count': 0,
                        'top_score': 0,
                        'avg_score': 0,
                        'high_relevance': 0
                    })

            except Exception as e:
                print(f"Error testing query '{query}': {e}")

    # Calculate metrics
    recall_rate = total_results_found / (total_queries * top_k) if total_queries > 0 else 0
    queries_with_results = sum(
        1 for cat_results in results_by_category.values()
        for r in cat_results if r['count'] > 0
    )
    coverage = queries_with_results / total_queries if total_queries > 0 else 0

    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    return {
        'threshold': threshold,
        'total_queries': total_queries,
        'queries_with_results': queries_with_results,
        'coverage': coverage,  # % of queries that returned at least one result
        'total_results': total_results_found,
        'avg_results_per_query': total_results_found / total_queries if total_queries > 0 else 0,
        'recall_rate': recall_rate,
        'avg_similarity_score': avg_similarity,
        'high_relevance_count': total_high_relevance,
        'results_by_category': dict(results_by_category)
    }


async def run_threshold_analysis():
    """Run comprehensive threshold analysis."""
    print("=" * 80)
    print("RAG SIMILARITY THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 80)

    qdrant = get_qdrant_service()

    # Verify connection
    print("\n[1/3] Verifying Qdrant connection...")
    info = qdrant.get_collection_info()
    points_count = info.get('points_count', 0)
    print(f"    ✓ Connected to collection: {info.get('name')}")
    print(f"    ✓ Embeddings available: {points_count}")
    print(f"    ✓ Vector dimension: {info.get('vector_size')}")

    if points_count == 0:
        print("    ✗ ERROR: No embeddings found. Please index lessons first.")
        return

    # Test each threshold
    print(f"\n[2/3] Testing {len(THRESHOLDS_TO_TEST)} threshold values...")
    print(f"    Query categories: {len(TEST_QUERIES)}")
    print(f"    Total test queries: {sum(len(q) for q in TEST_QUERIES.values())}")
    print()

    all_results = []

    for threshold in THRESHOLDS_TO_TEST:
        print(f"    Testing threshold {threshold:.2f}...", end=" ")
        result = await test_threshold_performance(qdrant, threshold)
        all_results.append(result)
        print(f"Coverage: {result['coverage']*100:.1f}%, Avg Score: {result['avg_similarity_score']:.3f}")

    # Analysis and recommendations
    print("\n[3/3] Analyzing results and generating recommendations...")
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    print(f"\n{'Threshold':<12} {'Coverage':<12} {'Avg Results':<14} {'Avg Score':<12} {'High Rel.':<12}")
    print("-" * 80)

    for result in all_results:
        print(f"{result['threshold']:<12.2f} "
              f"{result['coverage']*100:<11.1f}% "
              f"{result['avg_results_per_query']:<14.2f} "
              f"{result['avg_similarity_score']:<12.3f} "
              f"{result['high_relevance_count']:<12}")

    # Find optimal thresholds
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    # Best for coverage (max queries with results)
    best_coverage = max(all_results, key=lambda x: x['coverage'])

    # Best balance (coverage > 80% and highest avg similarity)
    balanced_candidates = [r for r in all_results if r['coverage'] >= 0.8]
    if balanced_candidates:
        best_balanced = max(balanced_candidates, key=lambda x: x['avg_similarity_score'])
    else:
        best_balanced = best_coverage

    # Best precision (highest avg similarity with reasonable coverage)
    precision_candidates = [r for r in all_results if r['coverage'] >= 0.7]
    if precision_candidates:
        best_precision = max(precision_candidates, key=lambda x: x['avg_similarity_score'])
    else:
        best_precision = best_balanced

    print("\n1. MAXIMUM COVERAGE (Best for general queries)")
    print(f"   Threshold: {best_coverage['threshold']:.2f}")
    print(f"   Coverage: {best_coverage['coverage']*100:.1f}%")
    print(f"   Avg Similarity: {best_coverage['avg_similarity_score']:.3f}")
    print(f"   Use case: General book questions, exploratory queries")
    print(f"   Trade-off: May return some less relevant results")

    print("\n2. BALANCED (Recommended for most use cases)")
    print(f"   Threshold: {best_balanced['threshold']:.2f}")
    print(f"   Coverage: {best_balanced['coverage']*100:.1f}%")
    print(f"   Avg Similarity: {best_balanced['avg_similarity_score']:.3f}")
    print(f"   Use case: Standard ChatKit queries, mixed question types")
    print(f"   Trade-off: Good balance between recall and precision")

    print("\n3. HIGH PRECISION (Best for specific technical queries)")
    print(f"   Threshold: {best_precision['threshold']:.2f}")
    print(f"   Coverage: {best_precision['coverage']*100:.1f}%")
    print(f"   Avg Similarity: {best_precision['avg_similarity_score']:.3f}")
    print(f"   Use case: Technical questions, specific topic queries")
    print(f"   Trade-off: May miss some general queries")

    # Category-specific analysis
    print("\n" + "=" * 80)
    print("CATEGORY-SPECIFIC PERFORMANCE")
    print("=" * 80)

    # Analyze performance by category at recommended threshold
    recommended_result = best_balanced

    for category, category_results in recommended_result['results_by_category'].items():
        queries_with_results = sum(1 for r in category_results if r['count'] > 0)
        avg_score = sum(r['top_score'] for r in category_results) / len(category_results) if category_results else 0

        print(f"\n{category}:")
        print(f"  Coverage: {queries_with_results}/{len(category_results)} queries")
        print(f"  Avg Top Score: {avg_score:.3f}")

        # Show queries without results
        no_results = [r['query'] for r in category_results if r['count'] == 0]
        if no_results:
            print(f"  ⚠ Queries without results:")
            for q in no_results:
                print(f"    - {q}")

    # Final recommendations
    print("\n" + "=" * 80)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 80)

    print(f"""
Current Configuration Analysis:
  - ChatKit (chatkit.py:98): similarity_threshold=0.35
  - Config default (config.py:67): rag_similarity_threshold=0.7
  - Test scripts (test_rag_search.py:37): similarity_threshold=0.3
  - Qdrant service (qdrant_client.py:195): similarity_threshold=0.5 (default)

Recommended Changes:

1. PRIMARY RECOMMENDATION - Update ChatKit endpoint:
   File: src/api/v1/chatkit.py:98
   Change: similarity_threshold=0.35 → similarity_threshold={best_balanced['threshold']:.2f}
   Reason: Optimal balance for educational content

2. Update configuration default:
   File: src/config.py:67
   Change: rag_similarity_threshold=0.7 → rag_similarity_threshold={best_balanced['threshold']:.2f}
   Reason: Align with empirically determined optimal value

3. Update service default:
   File: src/services/qdrant_client.py:195
   Change: similarity_threshold=0.5 → similarity_threshold={best_balanced['threshold']:.2f}
   Reason: Consistent defaults across the system

4. OPTIONAL - Dynamic threshold by query type:
   Consider implementing adaptive thresholds:
   - General queries (e.g., "what is the book about"): {best_coverage['threshold']:.2f}
   - Technical queries (e.g., "explain ROS2"): {best_precision['threshold']:.2f}
   - Default: {best_balanced['threshold']:.2f}

Trade-off Analysis:
  Precision vs Recall:
    - Lower threshold ({best_coverage['threshold']:.2f}): Higher recall, may include marginal results
    - Balanced threshold ({best_balanced['threshold']:.2f}): Good precision, good recall
    - Higher threshold ({best_precision['threshold']:.2f}): Higher precision, may miss some results

  Speed vs Accuracy:
    - Lower threshold returns more results → slightly more processing
    - Impact is minimal (top_k=5 limit applies)

  Cost:
    - No significant cost difference (same API calls for embeddings)
    - Marginal difference in LLM context size

Testing Instructions:
  1. Backup current settings
  2. Update threshold to {best_balanced['threshold']:.2f}
  3. Test with sample queries:
     python test_rag_search.py
  4. Monitor ChatKit responses for relevance
  5. Adjust if needed based on user feedback

Rollback Plan:
  If issues occur, revert to:
  - ChatKit: similarity_threshold=0.35
  - Config: rag_similarity_threshold=0.7
""")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_threshold_analysis())
