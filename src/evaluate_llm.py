"""
LLM evaluation: run the ground-truth ESI queries through multiple retrieval
approaches via the shared pipeline (each query logged once to llm_calls),
then read the logged calls back from the db, join them against the ground
truth in memory, and report which approach performs best. See llm_calls
table in db.py.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

import db
from pipeline import answer_query, build_rag

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"

# Retrieval methods to compare as the RAG context source, using the same prompt/model.
APPROACHES = [
    "hybrid_search_text_and_vector_medical",
    "vector_search_medical",
    "text_search",
]


def load_ground_truth() -> List[Dict]:
    """Load ground-truth {query, correct_answer} pairs for LLM evaluation."""
    content = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    return content["llm_queries"]


def summarize_approach(conn, batch_id: str, approach: str, ground_truth: Dict[str, int]) -> Dict:
    """Compute accuracy/answer rate for one approach by joining logged calls against ground truth."""
    rows = [row for row in db.load_llm_calls(conn, batch_id=batch_id) if row["approach"] == approach]
    total = len(rows)
    correct = sum(
        1 for row in rows
        if row["query"] in ground_truth and row["predicted_answer"] == ground_truth[row["query"]]
    )
    answered = sum(row["answered"] for row in rows)
    return {
        "approach": approach,
        "accuracy": correct / total if total else 0.0,
        "answer_rate": answered / total if total else 0.0,
    }


def main():
    queries = load_ground_truth()
    ground_truth = {item["query"]: item["correct_answer"] for item in queries}
    rag = build_rag()
    batch_id = str(uuid.uuid4())

    # Retrieve once per query and reuse across approaches: the approaches only differ
    # in which retrieved method's chunks are used as context, not in the retrieval itself.
    retrieval_cache = {item["query"]: rag.search(item["query"]) for item in queries}

    for approach in APPROACHES:
        for item in queries:
            answer_query(
                rag,
                item["query"],
                approach=approach,
                retrieval_method=approach,
                batch_id=batch_id,
                retrieval_results=retrieval_cache[item["query"]],
            )

    with db.get_connection() as conn:
        summaries = [summarize_approach(conn, batch_id, approach, ground_truth) for approach in APPROACHES]

    print(f"\nLLM evaluation batch {batch_id}")
    for summary in sorted(summaries, key=lambda s: s["accuracy"], reverse=True):
        print(f"  {summary['approach']}: accuracy={summary['accuracy']:.2%}  answer_rate={summary['answer_rate']:.2%}")

    best = max(summaries, key=lambda s: s["accuracy"])
    print(f"\nBest approach: {best['approach']} (accuracy={best['accuracy']:.2%})")


if __name__ == "__main__":
    main()
