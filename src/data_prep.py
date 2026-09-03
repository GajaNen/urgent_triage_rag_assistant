"""
Data preparation: Extract, chunk, and index PDFs for retrieval.
Supports text search and vector search with multiple embeddings.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

# Config
KB_DIR = Path(__file__).parent.parent / "knowledge_base"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

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

    def load_pdfs(self) -> List[Document]:
        """Extract text from all PDFs in knowledge base."""
        print(f"📂 Loading PDFs from {KB_DIR}")
        pdf_files = list(KB_DIR.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs")

        docs = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pdf_docs = loader.load()
                # Add source metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = pdf_path.name
                docs.extend(pdf_docs)
                print(f"✓ Loaded {len(pdf_docs)} pages from {pdf_path.name}")
            except Exception as e:
                print(f"✗ Error loading {pdf_path.name}: {e}")

        self.documents = docs
        return docs

    def chunk_documents(self) -> List[Document]:
        """Split documents into chunks."""
        print(f"\n🔀 Chunking {len(self.documents)} documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        chunks = self.splitter.split_documents(self.documents)
        print(f"Created {len(chunks)} chunks")

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks

    def create_embeddings(self, chunks: List[Document]) -> Dict[str, FAISS]:
        """Create vector stores with different embedding models."""
        print("\n🔤 Creating embeddings with multiple models")

        for model_name, model_path in EMBEDDING_MODELS.items():
            try:
                print(f"  → {model_name}: {model_path}")
                embeddings = HuggingFaceEmbeddings(model_name=model_path)
                vector_store = FAISS.from_documents(chunks, embeddings)
                self.vector_stores[model_name] = vector_store
                print(f"    ✓ Created FAISS index ({len(chunks)} chunks)")
            except Exception as e:
                print(f"    ✗ Error with {model_name}: {e}")

        return self.vector_stores

    def build_text_index(self, chunks: List[Document]) -> Dict[str, List[int]]:
        """Build simple keyword-based text index."""
        print("\n📝 Building text search index")
        text_index = {}

        for chunk in chunks:
            # Simple tokenization: split and lowercase
            tokens = chunk.page_content.lower().split()
            for token in tokens:
                if len(token) > 3:  # Skip short words
                    if token not in text_index:
                        text_index[token] = []
                    text_index[token].append(chunk.metadata["chunk_id"])

        print(f"✓ Text index has {len(text_index)} keywords")
        return text_index

    def save_data(self, chunks: List[Document], text_index: Dict):
        """Save chunks and indices to disk."""
        print("\n💾 Saving data")

        # Save chunks as JSON
        chunks_data = [
            {
                "id": c.metadata["chunk_id"],
                "content": c.page_content,
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", 0)
            }
            for c in chunks
        ]
        chunks_path = OUTPUT_DIR / "chunks.json"
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2)
        print(f"✓ Saved {len(chunks_data)} chunks to {chunks_path}")

        # Save text index
        text_index_path = OUTPUT_DIR / "text_index.json"
        with open(text_index_path, "w") as f:
            json.dump(text_index, f)
        print(f"✓ Saved text index to {text_index_path}")

        # Save vector stores
        for model_name, vs in self.vector_stores.items():
            vs_path = OUTPUT_DIR / f"vector_store_{model_name}"
            vs.save_local(str(vs_path))
            print(f"✓ Saved {model_name} vector store to {vs_path}")

    def run(self):
        """Execute full pipeline."""
        docs = self.load_pdfs()
        chunks = self.chunk_documents()
        text_index = self.build_text_index(chunks)
        self.create_embeddings(chunks)
        self.save_data(chunks, text_index)
        print("\n✅ Data prep complete")


if __name__ == "__main__":
    prep = DataPrep()
    prep.run()
