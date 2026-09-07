"""
Shared query-execution pipeline used by every entry point (api.py for live
traffic, evaluate_llm.py for ground-truth runs): build the RAG components
once per process, run a single query end-to-end (retrieve -> prompt -> LLM
-> parse -> log to llm_calls), and hand back the result. RAGBase itself
stays free of any db/evaluation concerns; this module is where those live.
"""

import time
import uuid
from typing import Any, Dict, Optional

from openai import OpenAI

import db
from rag import RAGBase
from retrieval import Retriever

# USD per 1M tokens. Update if pricing changes.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def build_rag(retrieval_method: str = "hybrid_search_text_and_vector_medical") -> RAGBase:
    """Construct the one Retriever + RAGBase instance a process should reuse for every query."""
    retriever = Retriever(queries=[], ncbi_search=True)
    client = OpenAI()
    return RAGBase(index=retriever, llm_client=client, retrieval_method=retrieval_method)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Estimate USD cost for a call from token counts, or None if pricing is unknown."""
    rates = PRICING.get(model)
    if not rates:
        return None
    return (prompt_tokens / 1_000_000) * rates["input"] + (completion_tokens / 1_000_000) * rates["output"]


def answer_query(
    rag: RAGBase,
    query: str,
    approach: Optional[str] = None,
    retrieval_method: Optional[str] = None,
    batch_id: Optional[str] = None,
    retrieval_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one query through RAG, log cost/tokens/latency/answered to the db, and return the result.

    `retrieval_method` picks which retrieved context to use (defaults to rag's fixed
    default) without mutating the shared `rag` instance. `approach` is just the label
    stored alongside the result for comparing runs (e.g. "production" vs a method name);
    it defaults to `retrieval_method` so evaluation batches can pass one value for both.
    Correctness against ground truth is computed later by joining logged `predicted_answer`
    values against the ground-truth file, not stored here.
    """
    approach = approach or retrieval_method or rag.retrieval_method
    batch_id = batch_id or str(uuid.uuid4())

    start = time.perf_counter()
    if retrieval_results is None:
        retrieval_results = rag.search(query)
    prompt = rag.build_prompt(query, retrieval_results, retrieval_method=retrieval_method)
    response = rag.llm(prompt)
    latency = time.perf_counter() - start

    assessment = response.output_parsed
    predicted = assessment.esi_level
    answered = predicted is not None
    answer = f"ESI is {predicted}. {assessment.rationale}" if answered else assessment.rationale

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage else None
    cost = (
        estimate_cost(rag.model, prompt_tokens, completion_tokens)
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )

    with db.get_connection() as conn:
        call_id = db.log_llm_call(
            conn,
            batch_id=batch_id,
            approach=approach,
            query=query,
            answered=answered,
            latency_seconds=latency,
            model=rag.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            predicted_answer=predicted,
            answer=answer,
        )

    return {
        "call_id": call_id,
        "batch_id": batch_id,
        "answer": answer,
        "esi_level": predicted,
        "latency_seconds": latency,
    }
