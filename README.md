# Urgent Triage RAG Assistant

A RAG-based urgent care triage assistant, built for the LLM Zoomcamp course.

## The problem

In an emergency department, a triage nurse has only a couple of minutes to
look at a patient's symptoms and vital signs and assign an **Emergency
Severity Index (ESI)** level from 1 (resuscitate immediately) to 5 (least
urgent) — a decision that determines how quickly that patient gets seen.
The official ESI handbook and related guidelines run to dozens of pages of
rules, decision trees, and edge cases, which is a lot to hold in your head
under time pressure.

This project is a RAG (Retrieval-Augmented Generation) assistant that takes
a free-text patient presentation (symptoms, vitals, history) and:
1. retrieves the most relevant passages from the ESI handbook and related
   triage manuals for that presentation, and
2. asks an LLM to read those passages and return a structured ESI level
   (1-5) plus a rationale citing the source material.

It's meant as a decision-support aid, not a replacement for clinical
judgement.

## How it works (example)

**Input** (`POST /query`):
> A 28-year-old patient presents with generalized abdominal pain. Her last
> menstrual period is reported as 8 weeks ago. Vital signs are as follows:
> T 36.7°C (98°F), HR 120 beats/minute, RR 22 breaths/minute, and BP
> 92/50mm Hg.

**Output:**
> ESI is 2. Her increased heart rate, decreased respiratory rate... wait —
> her increased heart rate, elevated respiratory rate, and decreased blood
> pressure make her high risk. This presentation could indicate internal
> bleeding from a ruptured ectopic pregnancy (ESI Handbook, Decision Point
> B, "high-risk situation" list).

Behind the scenes: the query is embedded/keyword-searched against the
knowledge base, the top-matching passages are put in the LLM's context, and
the LLM is instructed to only answer using that context (or say "I don't
know" / decline off-topic questions). See [docs/pipeline.md](docs/pipeline.md)
for the full flow.

## Overview

### Architecture
- **Knowledge Base**: Medical triage manuals (PDFs) and handbooks → extracted & chunked
- **Retrieval**: Hybrid approach comparing text search, vector search (general + medical embeddings) and hybrid search
- **LLM**: GPT-4o-mini (configurable) for triage reasoning, with structured (ESI level + rationale) output
- **Interface**: FastAPI (backend) + Streamlit (monitoring dashboard)
- **Orchestration**: dlt for the data ingestion pipeline

See [docs/pipeline.md](docs/pipeline.md) for a diagram, plain-English
description, and tool breakdown of each phase (ingestion, data prep,
retrieval, retrieval evaluation, RAG, RAG evaluation), including evaluation
results.

### Evaluation Strategy
- **Ground Truth**: hand-labeled triage queries (`data/test_queries.json`), based on NCLEX-style emergency nursing exam questions
- **Metrics**: retrieval hit rate/MRR, and LLM triage-level accuracy/answer rate
- **Best Practices**: hybrid search (reciprocal rank fusion), comparing multiple embedding models, structured LLM output

## Evaluation Criteria

For reviewers grading against the LLM Zoomcamp rubric, here's where each
criterion is addressed:

| Criterion | Where to look |
|---|---|
| Problem description | This README (above) |
| Retrieval flow (knowledge base + LLM) | [docs/pipeline.md](docs/pipeline.md#3-retrieval) (retrieval), [#5](docs/pipeline.md#5-rag) (RAG) |
| Retrieval evaluation (multiple approaches) | [docs/pipeline.md §4](docs/pipeline.md#4-retrieval-evaluation) + [§7 results](docs/pipeline.md#7-retrieval-evaluation-results) — 7 retrieval methods compared |
| LLM evaluation (multiple approaches) | [docs/pipeline.md §6](docs/pipeline.md#6-rag-evaluation) + [§8 results](docs/pipeline.md#8-rag-evaluation-results) — multiple models x retrieval approaches compared |
| Interface | FastAPI ([src/api.py](src/api.py)) + Streamlit dashboard ([src/dashboard.py](src/dashboard.py)) |
| Ingestion pipeline (automated) | [docs/pipeline.md §1](docs/pipeline.md#1-ingestion) — dlt-based, change-detecting |
| Monitoring | Streamlit dashboard + user feedback logging (`feedback` table, see [src/db.py](src/db.py)) |
| Containerization | `docker-compose.yml` — all services (ingest, api, dashboard, eval) run through it |
| Reproducibility | [docs/setup.md](docs/setup.md), [docs/usage.md](docs/usage.md), pinned `uv.lock` |
| Best practices | Hybrid search ✅ (see §3); document re-ranking ❌; query rewriting ❌ |

## Demo

*TODO: add a screenshot of the Streamlit dashboard and/or a short screen
recording of a query going through the API/dashboard here once the UI is
finalized.*

## Quick Start

The fastest way to run everything (ingestion, API, dashboard) is Docker —
see [docs/usage.md](docs/usage.md). Summary:

```bash
cp .env.example .env
# edit .env with your OPENAI_API_KEY (NCBI_API_KEY is optional)
docker compose up --build
```

This gives you the API at http://localhost:8000/docs and the dashboard at
http://localhost:8501. See `.env.example` for all supported environment
variables.

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

