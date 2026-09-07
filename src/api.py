"""
FastAPI service exposing the triage RAG pipeline: submit a query, get an ESI
assessment, and leave feedback. Every query logs cost/tokens/latency/answered
metrics to llm_calls (approach="production") for the dashboard.
"""

import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

import db
from pipeline import build_rag

load_dotenv()

app = FastAPI(title="Urgent Triage RAG Assistant")

# One shared Retriever + RAGBase for the process's lifetime; rebuilding per request
# would reload the vector stores and embedding models on every call.
_rag = build_rag()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query_id: int
    answer: str
    esi_level: Optional[int]
    latency_seconds: float


class FeedbackRequest(BaseModel):
    query_id: int
    reaction: str
    comment: Optional[str] = None


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    retrieval_results = _rag.search(request.query)
    prompt = _rag.build_prompt(request.query, retrieval_results)
    _rag.llm(prompt)
    latency_seconds = time.perf_counter() - start
    call_id = _rag.log_llm_call(
        request.query,
        approach="production",
        latency_seconds=latency_seconds,
    )
    return QueryResponse(
        query_id=call_id,
        answer=_rag.answer,
        esi_level=_rag.predicted_answer,
        latency_seconds=latency_seconds,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    with db.get_connection() as conn:
        db.log_feedback(
            conn,
            call_id=request.query_id,
            reaction=request.reaction,
            comment=request.comment,
        )
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
