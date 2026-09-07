# Pipeline

This project uses medical triage literature as knowledge base for a RAG assistant that
assigns an Emergency Severity Index (ESI 1-5) level to a described patient
presentation. It runs as six phases: ingest new source documents, prepare
(chunk + embed) them, retrieve relevant passages for a query using various methods, evaluate
retrieval quality, generate a triage answer with an LLM (RAG), and evaluate
that LLM output. Each phase below shows its flow and the tools used.


## Table of Contents

1. [Ingestion](#1-ingestion)
2. [Data Preparation](#2-data-preparation)
3. [Retrieval](#3-retrieval)
4. [Retrieval Evaluation](#4-retrieval-evaluation)
5. [RAG](#5-rag)
6. [RAG Evaluation](#6-rag-evaluation)
7. [Retrieval Evaluation Results](#7-retrieval-evaluation-results)
8. [RAG Evaluation Results](#8-rag-evaluation-results)
9. [API](#9-api)
10. [Dashboard](#10-dashboard)

---

## 1. Ingestion

```mermaid
flowchart TD
    A[Start: ingest_pipeline.py] --> B[dlt resource: kb_files\nscan knowledge_base/*.pdf]
    B --> C[dlt pipeline run\ndestination: duckdb state.duckdb]
    C --> D[Compute fingerprint\nsha256 of filename+mtime per PDF]
    D --> E{Fingerprint changed\nor force=True?}
    E -- No --> F[Skip: log 'no changes']
    E -- Yes --> G[DataPrep.run\nsee Data Preparation phase]
    G --> H[Write new fingerprint\nto last_ingested.txt]
```

Ingestion is automated with [dlt](https://dlthub.com/), a Python library for
building data pipelines. It watches the `knowledge_base/` folder and only
re-runs data preparation when a PDF is added, removed, or changed, so the
(slow) chunking/embedding step isn't repeated on every container start.

**Tools used:**
- **dlt** — orchestrates the pipeline: declares the `kb_files` resource, runs it, and persists state between runs.
- **DuckDB** — an embedded (file-based, no server) SQL database; used here purely as dlt's storage backend for the file-tracking table.
- **hashlib (stdlib)** — hashes the filename+modified-time of every PDF into one fingerprint, so a single comparison tells us if anything changed.

---

## 2. Data Preparation

```mermaid
flowchart TD
    A[Start: data_prep.py / DataPrep.run] --> B{redact=True?}
    B -- Yes --> C[Redact PDF regions\npymupdf add_redact_annot]
    B -- No --> D
    C --> D[Load PDFs\npymupdf extract text per page]
    D --> E[Chunk documents\nlangchain RecursiveCharacterTextSplitter\nsize=500, overlap=100]
    E --> F[Assign chunk_id metadata]
    F --> G[Create embeddings\nHuggingFace: general MiniLM + medical PubMedBERT]
    G --> H[Build FAISS vector stores\none per embedding model]
    F --> I[Save chunks + FTS5 index\nSQLite: chunks table]
    H --> J[Save vector stores to disk\ndata/vector_store_general, data/vector_store_medical]
    I --> K[Data prep complete]
    J --> K
```

Extracts text from the source PDFs, splits it into overlapping ~500-character
chunks (overlap=100), and indexes those chunks two ways: a keyword index (SQLite FTS5) and
two vector indices built with different embedding models, so retrieval
quality can be compared later.

**Tools used:**
- **PyMuPDF (`pymupdf`)** — reads PDF files and extracts their text/page layout; also draws the black-box redactions.
- **LangChain `RecursiveCharacterTextSplitter`** — splits long page text into overlapping ~500-character chunks along natural boundaries (paragraphs, sentences) so related content isn't cut mid-thought.
- **HuggingFace `sentence-transformers`** — turns each text chunk into a numeric vector (embedding) that captures its meaning; we compare a general-purpose model (MiniLM) against a medical-domain one (PubMedBERT).
- **FAISS** — a library for storing vectors and finding the nearest ones to a query vector quickly (vector similarity search).
- **SQLite + FTS5** — a lightweight, file-based SQL database; its FTS5 extension adds BM25-ranked keyword search with no extra service to run.

A local FAISS index is avoids running/paying for an extra service. Also didn't use ```Elasticsearch```/```minisearch```
for keyword search, since SQLite FTS5 gives BM25-style ranking with zero
extra infrastructure at this scale.

---

## 3. Retrieval

```mermaid
flowchart TD
    A[Query] --> B[Retriever.retrieve_query]
    B --> C[text_search\nSQLite FTS5 / BM25]
    B --> D[vector_search: general\nFAISS + MiniLM embeddings]
    B --> E[vector_search: medical\nFAISS + PubMedBERT embeddings]
    B --> F{ncbi_search enabled?}
    F -- Yes --> G[search_ncbi\nNCBI E-utilities esearch + efetch]
    F -- No --> H[skip]
    C --> I[hybrid_search\nReciprocal Rank Fusion]
    D --> I
    E --> I
    C --> J[Results per method]
    D --> J
    E --> J
    I --> J
    G --> J
    J --> K[Return dict: method -> ranked chunks/docs]
```

Given a query, runs every retrieval method in parallel (text, both vector
indices, and RRF-combined hybrids) so the outputs can be compared directly.
Optionally also queries PubMed live for supplementary evidence.

**Tools used:**
- **SQLite FTS5** — keyword/BM25 search over the chunk text (see Data Preparation).
- **FAISS + HuggingFace embeddings** — vector similarity search over the two embedding indices.
- **Reciprocal Rank Fusion (RRF, in-house)** — a simple formula (`1 / (rank + k)`) for merging ranked lists from different search methods into one combined ranking, without needing the raw scores to be on the same scale.
- **NCBI E-utilities API (`urllib`, stdlib)** — NCBI's free HTTP API for searching and fetching PubMed article abstracts, used for optional live supplementary evidence.

**Not used:** a re-ranker model (e.g. cross-encoder) — RRF over multiple
retrievers already gave a reasonable ranking signal for this corpus size, and
adding a re-ranker would mean another model to load/serve for limited
expected gain here.

---

## 4. Retrieval Evaluation

```mermaid
flowchart TD
    A[test_queries.json\nretrieval_queries ground truth] --> B[Retriever.retrieve_query\nfor every query, every method]
    B --> C[Save batch to SQLite\nretrieval_results table]
    C --> D[evaluate_retrieval.py]
    D --> E[compute_relevance_batch\nchunk_id match vs ground truth]
    E --> F[hit_rate per method]
    E --> G[mrr per method]
    C --> H[analyze_retrieval.py]
    H --> I[analyze_overlap\nJaccard similarity between methods]
    H --> J[summarize_performance\navg similarity score per method]
    H --> K[compare_sources\nsource distribution per method]
    F --> L[retrieval_analysis.json]
    G --> L
    I --> L
    J --> L
    K --> L
```

Runs a small hand-labeled set of triage queries (each mapped to the
`chunk_id` that should be retrieved) through every retrieval method, then
scores each method with hit rate and MRR, and separately checks how much the
methods agree with each other (Jaccard overlap) and which source documents
they favor.

**Tools used:**
- **Plain Python** — computes the metrics: hit rate (% of queries where the correct chunk appears in the results), MRR (mean reciprocal rank — rewards finding the correct chunk higher up the list), and Jaccard overlap (similarity between two methods' result sets).
- **SQLite** — stores the raw retrieval batch (every method's results for every query) so it can be re-analyzed without re-running retrieval.
- **JSON** — format for the ground-truth file (`test_queries.json`) and the final report (`retrieval_analysis.json`).

**Not used:** RAGAS or a similar off-the-shelf RAG-evaluation framework — the
ground truth here is chunk-id based (does the method retrieve the *known
correct* chunk), which a few lines of hit-rate/MRR code answer directly
without pulling in a framework built more for LLM-judged answer relevance.

---

## 5. RAG

```mermaid
flowchart TD
    A[User query] --> B[RAGBase.search\nRetriever.retrieve_query]
    B --> C[build_context\nselect retrieval_method results\n+ NCBI results if enabled]
    C --> D[build_prompt\nINSTRUCTIONS + PROMPT_TEMPLATE]
    D --> E[OpenAI chat completion\nstructured output: ESIAssessment]
    E --> F[Parse esi_level + rationale]
    F --> G[Compute tokens + cost\nPRICING table]
    F --> H[Log call to SQLite\nllm_calls table]
    G --> H
    H --> I[Return answer to caller\nAPI / dashboard]
```

Takes a free-text patient presentation, retrieves supporting context with one
configurable retrieval method, and asks an LLM to return a structured
ESI level (1-5) plus rationale, refusing to answer outside the triage domain.
Every call (tokens, cost, inputs/outputs) is logged for later evaluation and
monitoring.

**Tools used:**
- **OpenAI API** — the LLM call itself (default model: `gpt-4o-mini`).
- **Pydantic** (`ESIAssessment` model) — forces the LLM's response into a guaranteed shape (`esi_level` 1-5 or null, plus `rationale`), instead of parsing free-form text.
- **SQLite** — logs every call (query, model, tokens, cost, latency, predicted answer) for monitoring and later evaluation.

**Not used:** a full orchestration framework (LangChain chains/LangGraph) for
the RAG step itself — the flow is a single retrieve-then-generate call, so a
plain Python class is simpler to read and debug than wrapping it in a chain
abstraction.

---

## 6. RAG Evaluation

```mermaid
flowchart TD
    A[test_queries.json\nllm_queries ground truth] --> B[For each model in EVAL_MODELS]
    B --> C[For each retrieval approach\nhybrid / vector_medical / text]
    C --> D[RAGBase.rag per query\nsee RAG phase]
    D --> E[Call logged to\nllm_calls table]
    E --> F[summarize_run\naccuracy vs ground truth\nanswer_rate]
    F --> G[Aggregate summaries\nacross model x approach]
    G --> H[Report best combination\nby accuracy]
```

Replays a ground-truth set of `{query, correct_answer}` pairs through every
combination of LLM model and retrieval approach, then compares them on
accuracy (does the predicted ESI level match) and answer rate (did the model
actually assign a level instead of declining).

**Tools used:**
- **OpenAI models** — `gpt-4o-mini` and `gpt-4o` by default (configurable via `EVAL_MODELS`), compared head-to-head.
- **SQLite** — reads back the logged calls made during this evaluation run.
- **Plain Python** — computes accuracy (predicted ESI level == ground-truth level) and answer rate (% of queries where a level was assigned at all) per model/approach combination.

**Not used:** an LLM-as-judge for free-text answer quality — ESI level is a
discrete 1-5 label, so exact-match accuracy against ground truth is a more
direct and reproducible signal than asking another LLM to judge similarity. 

---

## 7. Retrieval Evaluation Results

From `data/retrieval_analysis.json` (11 ground-truth queries). *Numbers below
are from the current run in the repo — replace if you re-run the evaluation.*

| Method | Hit Rate | MRR |
|---|---|---|
| text_search | 0.182 | 0.121 |
| hybrid_search_text_and_vector_general | 0.091 | 0.091 |
| hybrid_search_text_and_vector_medical | 0.091 | 0.091 |
| hybrid_search_text_vector_general_and_medical | 0.091 | 0.091 |
| hybrid_search_vector_general_and_medical | 0.091 | 0.091 |
| vector_search_general | 0.091 | 0.018 |
| vector_search_medical | 0.091 | 0.091 |

Text search currently comes out ahead on both metrics, though all methods
score low in absolute terms — the ground-truth set is small (11 queries), so
these numbers should be treated as directional rather than conclusive. See
`data/retrieval_analysis.json` for per-method source distribution and
pairwise overlap between methods.

---

## 8. RAG Evaluation Results

Not run yet — `docker compose run eval` (or `python src/evaluate_llm.py`)
logs each call to `llm_calls` in the SQLite db and prints an accuracy /
answer-rate summary per model x retrieval approach, but doesn't persist that
summary to a file. Once a run is final, paste the printed summary table
here.

| Model | Retrieval approach | Accuracy | Answer rate |
|---|---|---|---|
| _pending_ | | | |

---

## 9. API

The FastAPI service exposes `/query` for RAG-based ESI assessments, `/feedback` for user reactions, and `/health` for Docker readiness checks.

---

## 10. Dashboard

The Streamlit dashboard sends patient presentations to the API, displays the generated ESI assessment, collects feedback, and visualizes query cost, token usage, latency, answer rate, and feedback.

---

## 2. Data Preparation

```mermaid
flowchart TD
    A[Start: data_prep.py / DataPrep.run] --> B{redact=True?}
    B -- Yes --> C[Redact PDF regions\npymupdf add_redact_annot]
    B -- No --> D
    C --> D[Load PDFs\npymupdf extract text per page]
    D --> E[Chunk documents\nlangchain RecursiveCharacterTextSplitter\nsize=500, overlap=100]
    E --> F[Assign chunk_id metadata]
    F --> G[Create embeddings\nHuggingFace: general MiniLM + medical PubMedBERT]
    G --> H[Build FAISS vector stores\none per embedding model]
    F --> I[Save chunks + FTS5 index\nSQLite: chunks table]
    H --> J[Save vector stores to disk\ndata/vector_store_general, data/vector_store_medical]
    I --> K[Data prep complete]
    J --> K
```

---

## 3. Retrieval

```mermaid
flowchart TD
    A[Query] --> B[Retriever.retrieve_query]
    B --> C[text_search\nSQLite FTS5 / BM25]
    B --> D[vector_search: general\nFAISS + MiniLM embeddings]
    B --> E[vector_search: medical\nFAISS + PubMedBERT embeddings]
    B --> F{ncbi_search enabled?}
    F -- Yes --> G[search_ncbi\nNCBI E-utilities esearch + efetch]
    F -- No --> H[skip]
    C --> I[hybrid_search\nReciprocal Rank Fusion]
    D --> I
    E --> I
    C --> J[Results per method]
    D --> J
    E --> J
    I --> J
    G --> J
    J --> K[Return dict: method -> ranked chunks/docs]
```

---

## 4. Retrieval Evaluation

```mermaid
flowchart TD
    A[test_queries.json\nretrieval_queries ground truth] --> B[Retriever.retrieve_query\nfor every query, every method]
    B --> C[Save batch to SQLite\nretrieval_results table]
    C --> D[evaluate_retrieval.py]
    D --> E[compute_relevance_batch\nchunk_id match vs ground truth]
    E --> F[hit_rate per method]
    E --> G[mrr per method]
    C --> H[analyze_retrieval.py]
    H --> I[analyze_overlap\nJaccard similarity between methods]
    H --> J[summarize_performance\navg similarity score per method]
    H --> K[compare_sources\nsource distribution per method]
    F --> L[retrieval_analysis.json]
    G --> L
    I --> L
    J --> L
    K --> L
```

---

## 5. RAG

```mermaid
flowchart TD
    A[User query] --> B[RAGBase.search\nRetriever.retrieve_query]
    B --> C[build_context\nselect retrieval_method results\n+ NCBI results if enabled]
    C --> D[build_prompt\nINSTRUCTIONS + PROMPT_TEMPLATE]
    D --> E[OpenAI chat completion\nstructured output: ESIAssessment]
    E --> F[Parse esi_level + rationale]
    F --> G[Compute tokens + cost\nPRICING table]
    F --> H[Log call to SQLite\nllm_calls table]
    G --> H
    H --> I[Return answer to caller\nAPI / dashboard]
```

---

## 6. RAG Evaluation

```mermaid
flowchart TD
    A[test_queries.json\nllm_queries ground truth] --> B[For each model in EVAL_MODELS]
    B --> C[For each retrieval approach\nhybrid / vector_medical / text]
    C --> D[RAGBase.rag per query\nsee RAG phase]
    D --> E[Call logged to\nllm_calls table]
    E --> F[summarize_run\naccuracy vs ground truth\nanswer_rate]
    F --> G[Aggregate summaries\nacross model x approach]
    G --> H[Report best combination\nby accuracy]
```

---

## 7. Retrieval Evaluation Results

TODO, blocked on final results.

---

## 8. RAG Evaluation Results

TODO, blocked on final results.

---

## 9. API

TODO.

---

## 10. Dashboard

TODO.
