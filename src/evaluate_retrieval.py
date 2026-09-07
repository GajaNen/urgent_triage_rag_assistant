"""
Retrieval evaluation: Compare performance of different retrieval methods.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import db

DATA_DIR = Path(__file__).parent.parent / "data"


def load_results() -> Dict:
    """Load the most recent retrieval evaluation batch from the database."""
    with db.get_connection() as conn:
        return db.load_retrieval_results(conn)

def load_ground_truth():
    """Load the ground truth (query, chunk_id) pairs from the json file."""
    json_content = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    return json_content["retrieval_queries"]


def analyze_overlap(results: Dict) -> Dict[str, Dict[str, float]]:
    """Calculate Jaccard similarity between retrieval methods."""
    analysis = {}

    for query, methods in results.items():
        if query not in analysis:
            analysis[query] = {}

        method_chunks = {method: [r["chunk_id"] for r in docs]
                         for method, docs in methods.items()
                         if method != "ncbi_search"}

        # Compare all method pairs
        method_names = list(set(method_chunks.keys()) - set(["ncbi_search"]))
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
        for method, retrieved_docs in methods.items():
            if method != "ncbi_search":
                methods_data[method].append({
                    "query": query,
                    # average score for this query and method
                    "avg_score_query": sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                })

    summary = {}
    for method, query_data in methods_data.items():
        summary[method] = {
            # average score for this method across all queries
            "avg_score_method": sum(query["avg_score_query"] for query in query_data) / len(query_data),
        }

    return summary


def compare_sources(results: Dict) -> Dict[str, Dict[str, int]]:
    """Analyze how many times each source is retrieved for each method."""
    source_counts = defaultdict(lambda: defaultdict(int))

    for query, methods in results.items():
        for method, docs in methods.items():
            if method != "ncbi_search":
                for doc in docs:
                    source = doc.get("source", "unknown")
                    source_counts[method][source] += 1

    return dict(source_counts)

# --------------------------------------------- #
# RELEVANCE

def compute_relevance(query_data: Dict[str, int], retrieved_results: List[Dict]):
    """Compute relevance of retrieved results for a single query based on chunk_id."""
    chunk_id = query_data["chunk_id"]

    relevance = []
    for result in retrieved_results:
        relevance.append(int(result["chunk_id"] == chunk_id))

    return relevance

def compute_relevance_batch(ground_truth: List[Dict], all_retrieved_results: Dict[Dict[str, List[Dict]]]):
    """Compute relevance for all queries in the ground truth for all methods."""
    relevance_batch: Dict[str, List[List[int]]] = defaultdict(list)

    for query_data in ground_truth:
        for method, retrieved_results in all_retrieved_results.get(query_data["query"], {}).items():
            if method != "ncbi_search":
                relevance = compute_relevance(query_data, retrieved_results)
                relevance_batch[method].append(relevance)

    return relevance_batch

def hit_rate(relevance):
    return sum([1 in line for line in relevance]) / len(relevance)

def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank, score in enumerate(line, 1):
            if score:
                total_score += 1 / rank
                break

    return total_score / len(relevance)


def print_report(results: Dict, ground_truth: List[Dict]):
    """Print comprehensive analysis report."""
    if not results:
        raise RuntimeError(
            "No retrieval results were found. Run ingestion and retrieval first, "
            "for example: docker compose up --build, then "
            "docker compose run --build --rm eval-retrieval."
        )

    print("\n" + "=" * 80)
    print("RETRIEVAL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Performance summary
    print("\nPERFORMANCE SUMMARY")
    print("-" * 80)
    summary = summarize_performance(results)
    for method in sorted(summary.keys()):
        metrics = summary[method]
        print(f"\n{method}")
        print(f"  Avg Score: {metrics['avg_score_method']:.3f}")

    # Source analysis
    print("\nSOURCE DOCUMENT DISTRIBUTION")
    print("-" * 80)
    sources = compare_sources(results)
    for method in sorted(sources.keys()):
        print(f"\n{method}")
        for source, count in sorted(sources[method].items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} results")

    # Method overlap
    print("\nMETHOD OVERLAP (Jaccard Similarity averaged over queries)")
    print("-" * 80)
    overlap = analyze_overlap(results)

    # Average overlap per comparison
    overlap_pairs = defaultdict(list)
    average_overlap_pairs = {}
    for query_data in overlap.values():
        for pair, jaccard in query_data.items():
            overlap_pairs[pair].append(jaccard)

    for pair in sorted(overlap_pairs.keys()):
        avg_jaccard = sum(overlap_pairs[pair]) / len(overlap_pairs[pair])
        average_overlap_pairs[pair] = avg_jaccard

    # print the pairs and their jaccard values in descending order of average Jaccard index
    for pair, jacc_idx in sorted(average_overlap_pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pair}: {jacc_idx:.3f}")

    # compute relevance for each method
    relevance_batch = compute_relevance_batch(ground_truth, results)
    if not relevance_batch:
        raise RuntimeError(
            "Retrieval results were loaded, but none matched the ground-truth queries. "
            "Rerun ingestion and retrieval so the database and vector stores are synchronized."
        )

    # compute and print overall relevance metrics
    print("\nRELEVANCE METRICS")
    print("-" * 80)
    for method, relevance in relevance_batch.items():
        print(f"\nMethod: {method}")
        print(f"  Hit Rate: {hit_rate(relevance):.3f}")
        print(f"  MRR: {mrr(relevance):.3f}")

    # Find best performing method
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    best_method = max(relevance_batch.items(), key=lambda x: mrr(x[1]))
    print(f"Best single method based on MRR: {best_method[0]}")
    print(f"  - MRR: {mrr(best_method[1]):.3f}")
    print("\n" + "=" * 80)


def save_report(results: Dict, ground_truth: List[Dict]):
    """Save analysis to file."""
    batch_relevance = compute_relevance_batch(ground_truth, results)
    report = {
        "summary": summarize_performance(results),
        "overlap": analyze_overlap(results),
        "sources": compare_sources(results),
        "relevance_batch": batch_relevance,
        "Hit Rate": {method: hit_rate(relevance) for method, relevance in batch_relevance.items()},
        "MRR": {method: mrr(relevance) for method, relevance in batch_relevance.items()}
    }

    report_path = DATA_DIR / "retrieval_analysis.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved analysis to {report_path}")


if __name__ == "__main__":
    ground_truth = load_ground_truth()
    #ground_truth_dict = ground_truth.to_dict(orient="query")
    #print(ground_truth)
    results = load_results()
    #print(results)
    print_report(results, ground_truth=ground_truth)
    save_report(results, ground_truth=ground_truth)
