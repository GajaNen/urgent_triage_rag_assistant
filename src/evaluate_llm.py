"""
LLM evaluation: run the ground-truth queries through multiple retrieval
approaches and models, then read the logged calls back
from the db, evaluate the responses against the ground truth in memory (json
file), and report which combination performs best. See llm_calls table in db.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

import db
from openai import OpenAI

from rag import RAGBase
from retrieval import Retriever

# load our api keys from the .env file
load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"

# Retrieval methods to compare as the RAG context source, using the same prompt/model.
APPROACHES = [
    "hybrid_search_text_and_vector_medical",
    "vector_search_medical",
    "text_search",
]

# Override with a comma-separated EVAL_MODELS value in .env or the shell.
MODELS = [model.strip() for model in os.getenv("EVAL_MODELS", "gpt-4o-mini,gpt-4o").split(",") if model.strip()]


def load_ground_truth() -> List[Dict]:
    """Load ground-truth {query, correct_answer} pairs for LLM evaluation."""
    content = json.loads((DATA_DIR / "test_queries.json").read_text(encoding="utf-8"))
    return content["llm_queries"]


def summarize_run(
    conn, call_ids: set[int], model: str, approach: str, ground_truth: Dict[str, int]
) -> Dict:
    """Compute accuracy/answer rate for one model and retrieval approach."""
    rows = [
        row
        for row in db.load_llm_calls(conn)
        if row["id"] in call_ids and row["model"] == model and row["approach"] == approach
    ]
    total = len(rows)
    correct = sum(
        1 for row in rows
        if row["query"] in ground_truth and row["predicted_answer"] == ground_truth[row["query"]]
    )
    answered = sum(row["answered"] for row in rows)
    return {
        "approach": approach,
        "model": model,
        "accuracy": correct / total if total else 0.0,
        "answer_rate": answered / total if total else 0.0,
    }


def main():
    queries = load_ground_truth() # a list of {query, correct_answer} dictionaries
    ground_truth = {item["query"]: item["correct_answer"] for item in queries} # dict of values

    llm_client = OpenAI()
    rags = {
        (model, approach): RAGBase(
            model=model,
            retrieval_method=approach,
            # we set ncbi_search to False to avoid live PubMed queries during evaluation
            # because it's too slow
            # the queries list is empty, but can be anything since it's only used
            # in a Retriever's method (called in evaluate_retrieval.py)
            # for batch retrieval for multiple queries
            # but it's never called in the RAGBase class which operates one query at a time
            retriever=Retriever(queries=[], ncbi_search=False),
            llm_client=llm_client,
        )
        for model in MODELS
        for approach in APPROACHES
    }
    call_ids_by_run = set()

    for model in MODELS:
        for approach in APPROACHES:
            rag = rags[(model, approach)]
            for item in queries:
                rag.rag(item["query"])
                with db.get_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT id
                        FROM llm_calls
                        WHERE model = ? AND approach = ? AND query = ?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (model, approach, item["query"]),
                    ).fetchone()
                if row is not None:
                    call_ids_by_run.add(row[0])

    with db.get_connection() as conn:
        summaries = [
            summarize_run(
                conn,
                call_ids_by_run,
                model,
                approach,
                ground_truth,
            )
            for model in MODELS
            for approach in APPROACHES
        ]

    print("\nLLM evaluation")
    for summary in sorted(summaries, key=lambda s: s["accuracy"], reverse=True):
        print(
            f"  {summary['model']} / {summary['approach']}: "
            f"accuracy={summary['accuracy']:.2%}  answer_rate={summary['answer_rate']:.2%}"
        )

    best = max(summaries, key=lambda s: s["accuracy"])
    print(
        f"\nBest combination: {best['model']} / {best['approach']} "
        f"(accuracy={best['accuracy']:.2%})"
    )


if __name__ == "__main__":
    main()
