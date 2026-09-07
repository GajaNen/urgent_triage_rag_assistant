"""
FastAPI service exposing the triage RAG pipeline: submit a query, get an ESI
assessment, and leave feedback. Every query logs cost/tokens/latency/answered
metrics to llm_calls (approach="production") for the dashboard.
"""

from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

import db
from pipeline import answer_query, build_rag

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
    result = answer_query(_rag, request.query, approach="production")
    return QueryResponse(
        query_id=result["call_id"],
        answer=result["answer"],
        esi_level=result["esi_level"],
        latency_seconds=result["latency_seconds"],
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
