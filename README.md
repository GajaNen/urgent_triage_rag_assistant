# Urgent Triage RAG Assistant

A RAG-based urgent care triage assistant for the LLM Zoomcamp course. Given a
free-text description of a patient's symptoms, it retrieves relevant passages
from medical triage handbooks and asks an LLM to assign an Emergency Severity
Index (ESI) level (1-5, 1 = most urgent) with a rationale.

## Overview

### Architecture
- **Knowledge Base**: Medical triage manuals (PDFs) and handbooks → extracted & chunked
- **Retrieval**: Hybrid approach comparing text search, vector search (general + medical embeddings) and hybrid search
- **LLM**: GPT-4o-mini (configurable) for triage reasoning, with structured (ESI level + rationale) output
- **Interface**: FastAPI (backend) + Streamlit (monitoring dashboard)
- **Orchestration**: dlt for the data ingestion pipeline

See [docs/pipeline.md](docs/pipeline.md) for a diagram and tool breakdown of
each phase (ingestion, data prep, retrieval, retrieval evaluation, RAG, RAG
evaluation), including evaluation results.

### Evaluation Strategy
- **Ground Truth**: hand-labeled triage queries (`data/test_queries.json`), based on NCLEX-style emergency nursing exam questions
- **Metrics**: retrieval hit rate/MRR, and LLM triage-level accuracy/answer rate
- **Best Practices**: hybrid search (reciprocal rank fusion), comparing multiple embedding models, structured LLM output

## Quick Start

The fastest way to run everything (ingestion, API, dashboard) is Docker —
see [docs/usage.md](docs/usage.md). Summary:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY (and NCBI_EMAIL/NCBI_API_KEY if using NCBI search)
docker compose up --build
```

This gives you the API at http://localhost:8000/docs and the dashboard at
http://localhost:8501.

To run the pieces manually instead (e.g. for development), see
[docs/setup.md](docs/setup.md).

## Project Structure

```
├── knowledge_base/          # Medical PDFs (triage manuals, ESI handbook)
├── src/
│   ├── ingest_pipeline.py  # dlt pipeline: detect changed PDFs, trigger data_prep
│   ├── data_prep.py        # Extract & chunk PDFs, create text + vector indices
│   ├── db.py                # SQLite schema/access (chunks, retrieval results, llm calls, feedback)
│   ├── retrieval.py        # Retrieval methods (text/vector/hybrid/NCBI)
│   ├── evaluate_retrieval.py # Retrieval hit rate / MRR vs ground truth
│   ├── analyze_retrieval.py  # Retrieval overlap/source analysis
│   ├── rag.py                # Retrieve + prompt + LLM call (RAGBase)
│   ├── evaluate_llm.py       # RAG accuracy/answer-rate across models & retrieval approaches
│   ├── api.py                # FastAPI backend
│   ├── dashboard.py          # Streamlit monitoring dashboard
│   └── ...
├── data/                    # Generated: app.db (SQLite), vector_store_general/, vector_store_medical/, retrieval_analysis.json
└── docs/
    ├── pipeline.md          # Phase-by-phase diagrams, tools, and results
    ├── setup.md             # Manual/local setup
    └── usage.md             # Running via Docker, sharing the app
```

## Retrieval Methods Compared

1. **Text Search** - Keyword-based lookup (fast, interpretable)
2. **Vector Search (General)** - General embeddings (sentence-transformers)
3. **Vector Search (Medical)** - Medical embeddings (PubMedBERT)
4. **Hybrid** - Reciprocal rank fusion of text + vector methods

Results and evaluation metrics are stored in the SQLite db (`data/app.db`)
and summarized in `data/retrieval_analysis.json` — see
[docs/pipeline.md](docs/pipeline.md#7-retrieval-evaluation-results) for the
current numbers.

## Next Steps

- [ ] Re-run retrieval/RAG evaluation on a larger ground-truth set
- [ ] Document the API and dashboard (docs/pipeline.md)
- [ ] Cloud deployment

