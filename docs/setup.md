# Setup Instructions

This covers running the pieces manually/locally (e.g. for development). For
the easiest way to run the whole app, see [usage.md](usage.md) (Docker).

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes the
  `docker compose` plugin) — only needed if you want to run the app via
  `docker compose` instead of running `api.py`/`dashboard.py` directly. On
  Windows/macOS, installing Docker Desktop is enough; on Linux, install the
  `docker-compose-plugin` package alongside the Docker Engine. Verify with:

  ```bash
  docker compose version
  ```

## Installation

```bash
# Install dependencies using uv
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows
```

You may need to run this command is your execution policy is set to ```Restricted```.

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Environment Setup

Copy the template and fill in your key:

```bash
cp .env.example .env
```

`OPENAI_API_KEY` is required (used by `rag.py`/`evaluate_llm.py`).
`NCBI_API_KEY` is optional (only used for the live PubMed lookup in
`retrieval.py`; it works without it too, just with lower rate limits). See
`.env.example` for the full list of supported variables.

## Data Preparation

Before running retrieval or the main application, extract and index the knowledge base:

```bash
python src/ingest_pipeline.py
```

This uses dlt to check whether any PDF in `knowledge_base/` is new/changed
since the last run; if so, it runs `DataPrep` (`src/data_prep.py`), which will:
- Extract text from all PDFs in `knowledge_base/`
- Split into chunks (size=500, overlap=100)
- Create vector indices with multiple embedding models:
  - General: `sentence-transformers/all-MiniLM-L6-v2`
  - Medical: `NeuML/pubmedbert-base-embeddings`
- Save processed data to `data/`:
  - `app.db` - SQLite db with the chunks table and an FTS5 keyword index
  - `vector_store_general/` - general embedding vectors (FAISS)
  - `vector_store_medical/` - medical embedding vectors (FAISS)

To force a full re-run even if nothing changed, call
`DataPrep().run()` directly (`python -c "from src.data_prep import DataPrep; DataPrep().run()"`)
or delete `data/dlt_pipeline/last_ingested.txt`.

**Note**: First run may take 5-10 minutes to download embedding models.

## Retrieval Evaluation

Evaluate and compare different retrieval methods against the hand-labeled
ground truth in `data/test_queries.json`:

```bash
python src/retrieval.py
python src/evaluate_retrieval.py
```

This compares:
1. **Text Search** - keyword-based retrieval (SQLite FTS5)
2. **Vector Search (General / Medical)** - using general-purpose vs. medical embeddings
3. **Hybrid Search variants** - combining text and vector results (reciprocal rank fusion)

Results are saved to the SQLite db and summarized in
`data/retrieval_analysis.json` (hit rate, MRR, overlap between methods). See
[pipeline.md](pipeline.md#7-retrieval-evaluation-results) for the current numbers.

## RAG Evaluation

Replay the ground-truth queries in `data/test_queries.json` through the RAG
pipeline across models/retrieval approaches:

```bash
python src/evaluate_llm.py
```

Prints an accuracy / answer-rate summary per model x retrieval approach; each
call is also logged to the SQLite db (`llm_calls` table).

## Output Structure

```
project/
├── data/
│   ├── app.db                       # SQLite: chunks, retrieval results, llm calls, feedback
│   ├── vector_store_general/        # General embeddings (FAISS)
│   ├── vector_store_medical/        # Medical embeddings (FAISS)
│   ├── test_queries.json            # Ground truth for retrieval + RAG evaluation
│   └── retrieval_analysis.json      # Retrieval evaluation summary
└── knowledge_base/
    ├── iitt_adult.pdf
    ├── iitt_pediatric.pdf
    ├── ESI-Handbook-5th-Edition-3-2023.pdf
    └── ...
```

