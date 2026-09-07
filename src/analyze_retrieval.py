"""
Retrieval analysis: Compare performance of different retrieval methods.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"


def load_results() -> Dict:
    """Load retrieval results."""
    results_path = DATA_DIR / "retrieval_results.json"
    with open(results_path) as f:
        return json.load(f)


def analyze_overlap(results: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate overlap between retrieval methods."""
    analysis = {}

    for query, methods in results.items():
        if query not in analysis:
            analysis[query] = {}

        method_chunks = {method: [r["chunk_id"] for r in docs]
                         for method, docs in methods.items()}

        # Compare all method pairs
        method_names = list(method_chunks.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[i + 1:]:
                chunks1 = set(method_chunks[method1])
                chunks2 = set(method_chunks[method2])
                overlap = len(chunks1 & chunks2)
                union = len(chunks1 | chunks2)
                jaccard = overlap / union if union > 0 else 0

                key = f"{method1} ↔ {method2}"
                if key not in analysis[query]:
                    analysis[query][key] = jaccard

    return analysis


def summarize_performance(results: Dict) -> Dict[str, Dict]:
    """Summarize performance metrics by method."""
    methods_data = defaultdict(list)

    for query, methods in results.items():
        for method, docs in methods.items():
            methods_data[method].append({
                "query": query,
                "num_results": len(docs),
                "avg_score": sum(d.get("score", 0) for d in docs) / len(docs) if docs else 0
            })

    summary = {}
    for method, data in methods_data.items():
        summary[method] = {
            "total_queries": len(data),
            "avg_results": sum(d["num_results"] for d in data) / len(data),
            "avg_score": sum(d["avg_score"] for d in data) / len(data),
            "result_variance": max([d["num_results"] for d in data]) - min([d["num_results"] for d in data])
        }

    return summary


def compare_sources(results: Dict) -> Dict[str, Dict[str, int]]:
    """Analyze which documents are being retrieved most."""
    source_counts = defaultdict(lambda: defaultdict(int))

    for query, methods in results.items():
        for method, docs in methods.items():
            for doc in docs:
                source = doc.get("source", "unknown")
                source_counts[method][source] += 1

    return dict(source_counts)


def print_report(results: Dict):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("RETRIEVAL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Performance summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("-" * 80)
    summary = summarize_performance(results)
    for method in sorted(summary.keys()):
        metrics = summary[method]
        print(f"\n{method}")
        print(f"  Queries: {metrics['total_queries']}")
        print(f"  Avg Results/Query: {metrics['avg_results']:.1f}")
        print(f"  Avg Score: {metrics['avg_score']:.3f}")
        print(f"  Result Variance: {metrics['result_variance']}")

    # Source analysis
    print("\n📁 SOURCE DOCUMENT DISTRIBUTION")
    print("-" * 80)
    sources = compare_sources(results)
    for method in sorted(sources.keys()):
        print(f"\n{method}")
        for source, count in sorted(sources[method].items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} results")

    # Method overlap
    print("\n🔄 METHOD OVERLAP (Jaccard Similarity)")
    print("-" * 80)
    overlap = analyze_overlap(results)

    # Average overlap per comparison
    overlap_pairs = defaultdict(list)
    for query_data in overlap.values():
        for pair, jaccard in query_data.items():
            overlap_pairs[pair].append(jaccard)

    for pair in sorted(overlap_pairs.keys()):
        avg_jaccard = sum(overlap_pairs[pair]) / len(overlap_pairs[pair])
        print(f"  {pair}: {avg_jaccard:.3f} (avg across queries)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)

    # Find best performing method
    best_method = max(summary.items(), key=lambda x: x[1]["avg_score"])
    print(f"✓ Best single method: {best_method[0]}")
    print(f"  - Avg Score: {best_method[1]['avg_score']:.3f}")

    # Hybrid search effectiveness
    if "hybrid_search" in summary:
        hybrid = summary["hybrid_search"]
        print(f"\n✓ Hybrid search performance:")
        print(f"  - Avg Score: {hybrid['avg_score']:.3f}")
        print(f"  - Good for combining strengths of text and vector search")

    print("\n" + "=" * 80)


def save_report(results: Dict):
    """Save analysis to file."""
    report = {
        "summary": summarize_performance(results),
        "overlap": analyze_overlap(results),
        "sources": compare_sources(results)
    }

    report_path = DATA_DIR / "retrieval_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved analysis to {report_path}")


if __name__ == "__main__":
    results = load_results()
    print_report(results)
    save_report(results)
