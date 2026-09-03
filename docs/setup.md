# Setup Instructions

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
# Install dependencies using uv
uv sync

# Activate the environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
NCBI_EMAIL=your.email@example.com
```

## Data Preparation

Before running retrieval or the main application, extract and index the knowledge base:

```bash
python src/data_prep.py
```

This will:
- Extract text from all PDFs in `knowledge_base/`
- Split into chunks (size=500, overlap=100)
- Create vector indices with multiple embedding models:
  - General: `sentence-transformers/all-MiniLM-L6-v2`
  - Medical: `NeuML/pubmedbert-base-embeddings`
- Build a keyword-based text search index
- Save processed data to `data/`:
  - `chunks.json` - document chunks
  - `text_index.json` - keyword index
  - `vector_store_general/` - general embedding vectors
  - `vector_store_medical/` - medical embedding vectors

**Note**: First run may take 5-10 minutes to download embeddings.

## Retrieval Evaluation

Evaluate and compare different retrieval methods:

```bash
python src/retrieval.py
```

This compares:
1. **Text Search** - keyword-based retrieval
2. **Vector Search (General)** - using general-purpose embeddings
3. **Vector Search (Medical)** - using medical-specific embeddings
4. **Hybrid Search** - combining text and vector (reciprocal rank fusion)

Results are saved to `data/retrieval_results.json` for analysis.

## Output Structure

```
project/
├── data/
│   ├── chunks.json                 # Document chunks
│   ├── text_index.json             # Keyword index
│   ├── vector_store_general/       # General embeddings
│   ├── vector_store_medical/       # Medical embeddings
│   └── retrieval_results.json      # Evaluation results
└── knowledge_base/
    ├── iitt_adult.pdf
    ├── iitt_pediatric.pdf
    ├── ESI-Handbook-5th-Edition-3-2023.pdf
    └── ...
```
