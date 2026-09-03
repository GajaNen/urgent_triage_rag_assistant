"""
Retrieval evaluation: Compare text search, vector search (multiple embeddings), and hybrid.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DATA_DIR = Path(__file__).parent.parent / "data"
KB_DIR = Path(__file__).parent.parent / "knowledge_base"

EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "medical": "NeuML/pubmedbert-base-embeddings"
}

# Test queries for evaluation
TEST_QUERIES = [
    "patient with chest pain and shortness of breath",
    "pediatric patient with high fever and rash",
    "elderly patient with severe headache and confusion",
    "patient with acute abdominal pain",
    "severe allergic reaction with breathing difficulty",
    "patient with fracture and heavy bleeding",
    "diabetic patient with altered mental status",
    "patient with severe burns"
]


class RetrieverEvaluator:
    def __init__(self):
        self.chunks = self.load_chunks()
        self.text_index = self.load_text_index()
        self.vector_stores = self.load_vector_stores()
        self.results = {}

    def load_chunks(self) -> List[Dict]:
        """Load preprocessed chunks."""
        chunks_path = DATA_DIR / "chunks.json"
        with open(chunks_path) as f:
            return json.load(f)

    def load_text_index(self) -> Dict[str, List[int]]:
        """Load text search index."""
        index_path = DATA_DIR / "text_index.json"
        with open(index_path) as f:
            return json.load(f)

    def load_vector_stores(self) -> Dict[str, FAISS]:
        """Load vector stores for all embedding models."""
        stores = {}
        for model_name in EMBEDDING_MODELS:
            try:
                vs_path = DATA_DIR / f"vector_store_{model_name}"
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[model_name])
                stores[model_name] = FAISS.load_local(str(vs_path), embeddings)
            except Exception as e:
                print(f"Warning: Could not load {model_name} vector store: {e}")
        return stores

    def text_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Simple text search using keyword index."""
        tokens = query.lower().split()
        chunk_scores = {}

        for token in tokens:
            if token in self.text_index:
                for chunk_id in self.text_index[token]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1

        # Sort by score descending
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:k]]

    def vector_search(self, query: str, model_name: str = "general", k: int = 5) -> List[Tuple[int, float]]:
        """Vector similarity search."""
        if model_name not in self.vector_stores:
            return []

        vs = self.vector_stores[model_name]
        results = vs.similarity_search_with_score(query, k=k)

        ranked = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is not None:
                ranked.append((chunk_id, score))

        return ranked

    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Combine text and vector search (reciprocal rank fusion)."""
        text_results = self.text_search(query, k=k * 2)
        vector_results = self.vector_search(query, model_name="medical", k=k * 2)

        # RRF: combine scores
        combined_scores = {}
        for rank, (chunk_id, score) in enumerate(text_results):
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + 1 / (rank + 60)

        for rank, (chunk_id, score) in enumerate(vector_results):
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + 1 / (rank + 60)

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def retrieve(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve using all methods."""
        results = {}

        # Text search
        text_results = self.text_search(query, k=k)
        results["text_search"] = [
            {
                "chunk_id": cid,
                "score": score,
                "content": self.chunks[cid]["content"][:200],
                "source": self.chunks[cid]["source"]
            }
            for cid, score in text_results
        ]

        # Vector search (multiple embeddings)
        for model_name in EMBEDDING_MODELS:
            if model_name in self.vector_stores:
                vector_results = self.vector_search(query, model_name=model_name, k=k)
                results[f"vector_search_{model_name}"] = [
                    {
                        "chunk_id": cid,
                        "score": float(score),
                        "content": self.chunks[cid]["content"][:200],
                        "source": self.chunks[cid]["source"]
                    }
                    for cid, score in vector_results
                ]

        # Hybrid search
        hybrid_results = self.hybrid_search(query, k=k)
        results["hybrid_search"] = [
            {
                "chunk_id": cid,
                "score": float(score),
                "content": self.chunks[cid]["content"][:200],
                "source": self.chunks[cid]["source"]
            }
            for cid, score in hybrid_results
        ]

        return results

    def evaluate_retrieval(self):
        """Run retrieval on test queries and save results."""
        print("🔍 Evaluating retrieval methods")
        print(f"  Test queries: {len(TEST_QUERIES)}")

        all_results = {}
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n  [{i}/{len(TEST_QUERIES)}] {query[:50]}...")
            results = self.retrieve(query, k=5)
            all_results[query] = results

            # Print summary
            for method, docs in results.items():
                print(f"    {method}: {len(docs)} results")

        self.results = all_results
        return all_results

    def save_results(self):
        """Save retrieval results to file."""
        output_path = DATA_DIR / "retrieval_results.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved retrieval results to {output_path}")

    def run(self):
        """Execute evaluation."""
        self.evaluate_retrieval()
        self.save_results()
        print("\n✅ Retrieval evaluation complete")


if __name__ == "__main__":
    evaluator = RetrieverEvaluator()
    evaluator.run()
