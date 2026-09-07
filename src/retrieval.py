"""
Retrieval evaluation: Compare text search, vector search (multiple embeddings), and hybrid.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os
import re
import urllib.request
import urllib.parse

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import db

DATA_DIR = Path(__file__).parent.parent / "data"
KB_DIR = Path(__file__).parent.parent / "knowledge_base"

EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "medical": "NeuML/pubmedbert-base-embeddings"
}

# load the test queries from the json file
TEST_QUERIES = {}
with open(DATA_DIR / "test_queries.json", "r", encoding="utf-8") as f:
    TEST_QUERIES = [item["query"] for item in json.load(f)["retrieval_queries"]]


class Retriever:
    """Retrieves data from the knowledge base using text, vector and hybrid
    search, optionally calling NCBI api to search online medical articles in real-time."""
    def __init__(
            self, 
            queries: List[str] = TEST_QUERIES, 
            ncbi_search: bool = False,
            ncbi_api_key: Optional[str] = os.getenv("NCBI_API_KEY")
        ):
        self.queries = queries
        self.ncbi_search = ncbi_search
        self.ncbi_api_key = ncbi_api_key
        self.chunks = self.load_chunks()
        self.vector_stores = self.load_vector_stores()
        self.results = {}
        

    def load_chunks(self) -> List[Dict]:
        """Load preprocessed chunks from SQLite."""
        with db.get_connection() as conn:
            return db.load_chunks(conn)

    def load_vector_stores(self) -> Dict[str, FAISS]:
        """Load vector stores for all embedding models."""
        stores = {}
        for model_name in EMBEDDING_MODELS:
            try:
                # load vector store for this model
                vs_path = DATA_DIR / f"vector_store_{model_name}"
                # determine embedding type for this model
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODELS[model_name])
                # retrieve vectors based on the saved index (of chunks) and type of emebdding
                stores[model_name] = FAISS.load_local(str(vs_path), embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Warning: Could not load {model_name} vector store: {e}")
       
        return stores

    def text_search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Full-text search using SQLite FTS5 (BM25 ranking)."""
        with db.get_connection() as conn:
            return db.search_text(conn, query, k=k)

    def vector_search(self, query: str, model_name: str = "general", k: int = 5) -> List[Tuple[int, float]]:
        """Vector similarity search."""
        #print(model_name, self.vector_stores)
        if model_name not in self.vector_stores:
            return []

        vs = self.vector_stores[model_name]
        # look it up: which method and what does the score entail exactly
        results = vs.similarity_search_with_score(query, k=k)

        ranked = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is not None:
                ranked.append((chunk_id, score))

        return ranked

    # this way this function can be used as a standalong
    # but maybe it could take text and vector results as input already
    # bc otherwise we do these two searches two times in the main retrieval function
    # also this uses only medical model

    # add another metric and then pool the scores so that we're not choosing
    # solely based on RRF)
    def hybrid_search(self, results_lists: List[List[Tuple[int, float]]], k: int=5, rrf_denom: int=60) -> List[Tuple[int, float]]:
        """Combine text and vector search (reciprocal rank fusion)."""
        # RRF: combine scores
        combined_scores: Dict[int, float] = {}
        for results in results_lists:
            #print([x for x in enumerate(results)])
            for rank, (chunk_id, score) in enumerate(results):
                combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + 1 / (rank + rrf_denom)

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def _extract_ncbi_keywords(self, query: str) -> str:
        """Clean natural language query into boolean medical search terms for PubMed."""
        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "are", "was", "were", "what", "how", "why", "who", "which",
            "patient", "patients", "presents", "presenting", "management", "treatment"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        keywords = [w for w in words if w not in stopwords]
        return " AND ".join(keywords[:5]) if keywords else query


    # add a function for api call
    # it would take keywords from the query and use them to search the NCBI API
    # then it would parse the API response and return the top k results:
    # but save the identifying info so that the contents can be retrieved later
    # it could also be used in hybrid search then 
    def search_ncbi(self, query: str, k: int = 5) -> List[Dict]:
        """Search PubMed via NCBI E-utilities API and fetch article abstracts."""
        search_terms = self._extract_ncbi_keywords(query)
        try:
            params = {
                "db": "pubmed",
                "term": search_terms,
                "retmode": "json",
                "retmax": k
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            # 1. Search PubMed for PMIDs
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(search_url, headers={"User-Agent": "UrgentTriageRAG/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                search_data = json.loads(resp.read().decode("utf-8"))

            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            # 2. Fetch Abstracts via efetch (XML)
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            if self.ncbi_api_key:
                fetch_params["api_key"] = self.ncbi_api_key

            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(fetch_params)
            req = urllib.request.Request(fetch_url, headers={"User-Agent": "UrgentTriageRAG/1.0"})
            
            import xml.etree.ElementTree as ET
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)

            results = []
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                title_elem = article.find(".//ArticleTitle")
                abstract_elems = article.findall(".//AbstractText")
                source_elem = article.find(".//Title")
                
                pmid = pmid_elem.text if pmid_elem is not None else ""
                title = title_elem.text if title_elem is not None else "No Title"
                source = source_elem.text if source_elem is not None else "PubMed"
                
                # Join abstract paragraphs
                abstract_text = " ".join([elem.text for elem in abstract_elems if elem.text])
                
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "source": f"PubMed: {source}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "content": f"Title: {title}\nAbstract: {abstract_text if abstract_text else 'No abstract available.'}"
                })
            return results
        except Exception as e:
            print(f"NCBI Search error for search terms '{search_terms}': {e}")
            return []
    

    def retrieve_query(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve using all methods."""
        results = {}
        all_vector_results = {}

        # Text search
        # we take twice as many results from each method to allow for better fusion
        # in the hybrid search, fusing k * 2 results, to eliminate some of random noise
        text_results = self.text_search(query, k=k * 2)
        # for actual text / vector results we take only top k
        results["text_search"] = [
            {
                "chunk_id": cid,
                "score": score,
                "content": self.chunks[cid]["content"],
                "source": self.chunks[cid]["source"]
            }
            for cid, score in text_results[:k]
        ]

        # Vector search (using multiple embeddings based on different models)
        for model_name in EMBEDDING_MODELS:
            #print("retrieving vector search results for model:", model_name, "available vector stores:", self.vector_stores.keys())
            if model_name in self.vector_stores:
                vector_results = self.vector_search(query, model_name=model_name, k=k * 2)
                all_vector_results[model_name] = vector_results
                #print("vector_results after adding model_name:", model_name, vector_results)
                results[f"vector_search_{model_name}"] = [
                    {
                        "chunk_id": cid,
                        "score": float(score),
                        "content": self.chunks[cid]["content"],
                        "source": self.chunks[cid]["source"]
                    }
                    for cid, score in vector_results[:k]
                ]

        # Hybrid search: merge text & vector search results using reciprocal rank fusion
        # merge vecg vecm, vecg text, vecm text, vecg vecm text

        vec_gen_res = all_vector_results.get("general", [])
        vec_med_res = all_vector_results.get("medical", [])

        hybrid_combos = {
            "text_and_vector_general": [text_results, vec_gen_res],
            "text_and_vector_medical": [text_results, vec_med_res],
            "vector_general_and_medical": [vec_gen_res, vec_med_res],
            "text_vector_general_and_medical": [text_results, vec_gen_res, vec_med_res],
        }
        
        for hybrid_name, hybrid_lists in hybrid_combos.items():
            # for hybrid search we also take only top k results based on top k * 2
            # of each individual method
            hybrid_results = self.hybrid_search(hybrid_lists, k=k)
            results[f"hybrid_search_{hybrid_name}"] = [
                {
                    "chunk_id": cid,
                    "score": float(score),
                    "content": self.chunks[cid]["content"],
                    "source": self.chunks[cid]["source"]
                }
                for cid, score in hybrid_results
            ]

        # NCBI live PubMed search
        if self.ncbi_search:
            ncbi_docs = self.search_ncbi(query, k=k)
            if ncbi_docs:
                results["ncbi_search"] = ncbi_docs

        return results

    def retrieve_batch(self):
        """Run retrieval on test queries and save results."""
        print("Performing retrieval using all methods")
        print(f"  Test queries: {len(self.queries)}")

        all_results = {}
        for i, query in enumerate(self.queries, 1):
            print(f"\n  [{i}/{len(self.queries)}] {query[:50]}...")
            results = self.retrieve_query(query, k=5)
            all_results[query] = results

            # Print summary
            for method, docs in results.items():
                print(f"    {method}: {len(docs)} results")

        self.results = all_results
        return all_results

    # save to db
    def save_results(self):
        """Save retrieval results to the database."""
        with db.get_connection() as conn:
            batch_id = db.save_retrieval_results(conn, self.results)
        print(f"\nSaved retrieval results to {db.DB_PATH} (batch {batch_id})")

    def run(self):
        """Execute retrieval for a batch of queries."""       
        self.retrieve_batch()
        self.save_results()
        print("\nRetrieval complete")


if __name__ == "__main__":
    # if you want NCBI live PubMed search, set ncbi_search=True
    retriever = Retriever()
    retriever.run()
