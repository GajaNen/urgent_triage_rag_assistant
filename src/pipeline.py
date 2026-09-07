"""Shared RAG construction helpers."""

from openai import OpenAI

from rag import RAGBase
from retrieval import Retriever
from openai import OpenAI

import dotenv

dotenv.load_dotenv()


def build_rag(
    retrieval_method: str = "hybrid_search_text_and_vector_medical",
    model: str = "gpt-4o-mini",
    retriever=None,
    llm_client=None,
) -> RAGBase:
    """Construct the one Retriever + RAGBase instance a process should reuse for every query."""
    if retriever is None:
        retriever = Retriever(queries=[], ncbi_search=False)
    if llm_client is None:
        llm_client = OpenAI()
    return RAGBase(
        retriever=retriever,
        llm_client=llm_client,
        retrieval_method=retrieval_method,
        model=model,
    )


