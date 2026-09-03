# Urgent Triage RAG Assistant

A RAG-based urgent care triage assistant for the LLM Zoomcamp course. Evaluates patient symptoms and assigns triage severity using medical knowledge base retrieval and LLM reasoning.

## Overview

### Architecture
- **Knowledge Base**: Medical triage manuals (PDFs) and handbooks → extracted & chunked
- **Retrieval**: Hybrid approach comparing text search, vector search (general + medical embeddings) and hybrid search
- **LLM**: GPT-4o-mini for keyword extraction and triage reasoning
- **Interface**: FastAPI (backend) + Streamlit (monitoring dashboard)
- **Orchestration**: dlt/Airflow for data ingestion pipeline

### Evaluation Strategy
- **Ground Truth**: NCLEX emergency nursing exams & triage practice tests
- **Metrics**: Retrieval quality, semantic similarity to reference answers, triage level accuracy
- **Best Practices**: Hybrid search, document re-ranking, query rewriting

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and NCBI_EMAIL
```

### 3. Prepare Data
```bash
python src/data_prep.py
```
Creates indexed knowledge base (vectors + text search)

### 4. Evaluate Retrieval
```bash
python src/retrieval.py
python src/analyze_retrieval.py
```
Compares retrieval methods and shows performance analysis

## Project Structure

```
├── knowledge_base/          # Medical PDFs (triage manuals, ESI handbook)
├── src/
│   ├── data_prep.py        # Extract & chunk PDFs, create indices
│   ├── retrieval.py        # Retrieval methods comparison
│   ├── analyze_retrieval.py # Performance analysis
│   ├── starter.py          # Example pipeline
│   └── ...
├── data/                    # Generated indices & results
│   ├── chunks.json
│   ├── text_index.json
│   ├── vector_store_general/
│   ├── vector_store_medical/
│   └── retrieval_results.json
└── docs/
    ├── setup.md            # Detailed setup & output structure
    └── usage.md
```

## Retrieval Methods Compared

1. **Text Search** - Keyword-based lookup (fast, interpretable)
2. **Vector Search (General)** - General embeddings (sentence-transformers)
3. **Vector Search (Medical)** - Medical embeddings (PubMedBERT)
4. **Hybrid** - Reciprocal rank fusion of text + medical vectors

Results saved to `data/retrieval_results.json` and summarized in `retrieval_analysis.json`.

## Next Steps

- [ ] RAG pipeline integration with LangChain
- [ ] FastAPI endpoint for triage
- [ ] Streamlit monitoring dashboard
- [ ] Evaluation against ground truth (nursing exams)
- [ ] Automated ingestion pipeline (dlt/Airflow)
- [ ] Containerization (Docker)
- [ ] Cloud deployment
