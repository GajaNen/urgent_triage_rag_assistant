"""
Data preparation: Extract, chunk, and index PDFs for retrieval.
Supports text search and vector search with multiple embeddings.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import pickle

from annotated_types import doc
import pymupdf  # PyMuPDF: fitz api is deprecated
import db
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# Config
KB_DIR = Path(__file__).parent.parent / "knowledge_base"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Redaction: the "_unredacted" file is the untouched source and is never indexed directly.
REDACT_SOURCE = KB_DIR / "ESI-Handbook-5th-Edition-3-2023_unredacted.pdf"
REDACT_TARGET = KB_DIR / "ESI-Handbook-5th-Edition-3-2023.pdf"
# 0-based page index -> list of (x0, y0, x1, y1) rects in PDF points, origin top-left.
# Page indices are offset by 5 from the handbook's printed page numbers.
REDACT_REGIONS = {
    25: [(30, 348, 582, 686)],    # printed p.20: all of Table 5-2, between Table 5-1 (ends y=342) and its footnotes (start y=691)
    29: [(306, 345, 584, 720)],   # printed p.24: right column, below Table 6-2 ("Case Examples".."Example Two")
    30: [(30, 60, 306, 720)],     # printed p.25: whole left column ("Example Three".."Example Five")
}

# Embedding models to compare
EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "medical": "NeuML/pubmedbert-base-embeddings"
}

class DataPrep:
    def __init__(self):
        self.documents = []
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_stores = {}

    def redact_pdf(
            self,
            source: Path = REDACT_SOURCE,
            target: Path = REDACT_TARGET,
            regions: Dict[int, List[Any]] = REDACT_REGIONS,
        ) -> Path:
        """Black out the configured regions, keeping the rest of the text layer intact."""
        print(f"\nRedacting {source.name} -> {target.name}")
        with pymupdf.open(source) as doc:
            for page_index, rects in regions.items():
                page = doc[page_index]
                for rect in rects:
                    page.add_redact_annot(pymupdf.Rect(rect), fill=(0, 0, 0))
                page.apply_redactions()
                print(f"  page {page_index}: redacted {len(rects)} region(s)")
            doc.save(target, garbage=4, deflate=True)
        return target

    def load_pdfs(self) -> List[Document]:
        """Extract text from all PDFs in knowledge base."""
        print(f"Loading PDFs from {KB_DIR}.")
        pdf_files = [p for p in KB_DIR.glob("*.pdf") if not p.stem.endswith("_unredacted")]
        print(f"Found {len(pdf_files)} PDFs.")

        docs = []
        for pdf_path in pdf_files:
            try:
                page_docs = []
                with pymupdf.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf):
                        text = page.get_text()
                        if not text.strip(): # omit empty pages
                            print("empty page, skipping", f"({pdf_path.name}, page {page_num})")
                            continue
                        # collect content and metadata of a page
                        page_docs.append(Document(
                            page_content=text,
                            metadata={"source": pdf_path.name, "page": page_num}
                        ))
                # add the page to the docs object
                docs.extend(page_docs)
                print(f"Loaded {len(page_docs)} pages from {pdf_path.name}")
            except Exception as e:
                print(f"Error loading {pdf_path.name}: {e}")

        self.documents = docs
        return docs

    def chunk_documents(self) -> List[Document]:
        """Split documents (pages) into chunks."""
        print(f"\nChunking {len(self.documents)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        chunks = self.splitter.split_documents(self.documents)
        print(f"Created {len(chunks)} chunks")

        # Add a unique chunk id to its metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks

    def create_embeddings(self, chunks: List[Document]) -> Dict[str, FAISS]:
        """Create vector stores with different embedding models."""
        print("\nCreating embeddings with multiple models")

        for model_name, model_path in EMBEDDING_MODELS.items():
            try:
                print(f"  → {model_name}: {model_path}")
                embeddings = HuggingFaceEmbeddings(model_name=model_path)
                vector_store = FAISS.from_documents(chunks, embeddings)
                self.vector_stores[model_name] = vector_store
                print(f"    Created FAISS index ({len(chunks)} chunks)")
            except Exception as e:
                print(f"    Error with {model_name}: {e}")

        return self.vector_stores


    def save_data(self, chunks: List[Document]):
        """Save chunks (and FTS5 index) to the SQLite database and vector stores to disk."""
        print("\nSaving data")

        chunks_data = [
            {
                "id": c.metadata["chunk_id"],
                "content": c.page_content,
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", 0)
            }
            for c in chunks
        ]
        with db.get_connection() as conn:
            db.save_chunks(conn, chunks_data)
        print(f"Saved {len(chunks_data)} chunks and FTS5 index to {db.DB_PATH}")

        # Save vector stores
        for model_name, vs in self.vector_stores.items():
            vs_path = OUTPUT_DIR / f"vector_store_{model_name}"
            vs.save_local(str(vs_path))
            print(f"Saved {model_name} vector store to {vs_path}")

    def run(self, redact: bool = False):
        """Execute full pipeline."""
        if redact:
            self.redact_pdf()
        docs = self.load_pdfs()
        chunks = self.chunk_documents()
        self.create_embeddings(chunks)
        self.save_data(chunks)
        print("\nData prep complete")


if __name__ == "__main__":
    prep = DataPrep()
    # if you want to redact the file yourself from the unredacted set redact=True
    # we already have the redacted version, so we're not redacting by default
    prep.run()
