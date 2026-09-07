"""
SQLite storage layer: chunks, full-text search index, LLM call logs (usage,
cost, latency for every query, production or evaluation), and user feedback
on individual calls.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT,
    page INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    batch_id TEXT NOT NULL,
    query TEXT NOT NULL,
    method TEXT NOT NULL,
    rank INTEGER NOT NULL,
    chunk_id INTEGER,
    score REAL,
    content TEXT,
    source TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    batch_id TEXT NOT NULL,
    approach TEXT NOT NULL,
    query TEXT NOT NULL,
    predicted_answer INTEGER,
    answered INTEGER NOT NULL,
    answer TEXT,
    latency_seconds REAL,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd REAL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    call_id INTEGER NOT NULL REFERENCES llm_calls(id),
    reaction TEXT NOT NULL,
    comment TEXT
);
"""


def _migrate_legacy_feedback_table(conn: sqlite3.Connection) -> None:
    """Rebuild `feedback` if it predates the `call_id` column (old dev DBs only had query/response)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()}
    if columns and "call_id" not in columns:
        conn.execute("DROP TABLE feedback")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with schema ensured, committing on success."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        _migrate_legacy_feedback_table(conn)
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_chunks(conn: sqlite3.Connection, chunks_data: List[Dict[str, Any]]) -> None:
    """Replace stored chunks and rebuild the FTS5 index from them."""
    # we want to delete the old chunks with every run because of potential changes
    # in our data prep procedure/input materials
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT INTO chunks (id, content, source, page) VALUES (:id, :content, :source, :page)",
        chunks_data,
    )
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")


def load_chunks(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Load all chunks as a list of dicts, ordered by id."""
    cur = conn.execute("SELECT id, content, source, page FROM chunks ORDER BY id")
    return [
        {"id": row[0], "content": row[1], "source": row[2], "page": row[3]}
        for row in cur.fetchall()
    ]


def search_text(conn: sqlite3.Connection, query: str, k: int = 5) -> List[Tuple[int, float]]:
    """Full-text search chunks via FTS5 BM25 ranking. Returns (chunk_id, score) with higher = better."""
    # tokenise the query: each word is one token
    tokens = query.lower().split()
    if not tokens:
        return []

    # Quote each token so FTS5 treats it as a literal term, not query syntax
    match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)

    # match a chunk agains any word of the input query and return first k results
    cur = conn.execute(
        """
        SELECT rowid, bm25(chunks_fts) AS rank
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (match_query, k),
    )
    # sqlite's bm25() is lower-is-better; negate so higher score = better match
    # maybe subtract from 100 instead of negating it
    return [(row[0], -row[1]) for row in cur.fetchall()]


def save_retrieval_results(conn: sqlite3.Connection, results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> str:
    """Append a retrieval evaluation run as rows tagged with a shared batch id. Returns the batch id."""
    batch_id = str(uuid.uuid4())
    rows = [
        (
            batch_id,
            query,
            method,
            rank,
            doc.get("chunk_id"),
            doc.get("score"),
            doc.get("content"),
            doc.get("source"),
        )
        for query, methods in results.items()
        for method, docs in methods.items()
        for rank, doc in enumerate(docs)
    ]
    conn.executemany(
        """
        INSERT INTO retrieval_results (batch_id, query, method, rank, chunk_id, score, content, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return batch_id


def load_retrieval_results(conn: sqlite3.Connection, batch_id: Optional[str] = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load a retrieval evaluation run as {query: {method: [docs]}}. Defaults to the most recent batch."""
    if batch_id is None:
        cur = conn.execute("SELECT batch_id FROM retrieval_results ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return {}
        batch_id = row[0]

    cur = conn.execute(
        """
        SELECT query, method, chunk_id, score, content, source
        FROM retrieval_results
        WHERE batch_id = ?
        ORDER BY query, method, rank
        """,
        (batch_id,),
    )
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for query, method, chunk_id, score, content, source in cur.fetchall():
        results.setdefault(query, {}).setdefault(method, []).append(
            {"chunk_id": chunk_id, "score": score, "content": content, "source": source}
        )
    return results


def log_llm_call(
    conn: sqlite3.Connection,
    batch_id: str,
    approach: str,
    query: str,
    answered: bool,
    latency_seconds: float,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    cost_usd: Optional[float] = None,
    predicted_answer: Optional[int] = None,
    answer: Optional[str] = None,
) -> int:
    """Record one LLM call (production or evaluation) with usage/cost/latency. Returns the row id."""
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    cur = conn.execute(
        """
        INSERT INTO llm_calls (
            batch_id, approach, query, predicted_answer, answered, answer,
            latency_seconds, model, prompt_tokens, completion_tokens, total_tokens, cost_usd
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id, approach, query, predicted_answer, int(answered), answer,
            latency_seconds, model, prompt_tokens, completion_tokens, total_tokens, cost_usd,
        ),
    )
    return cur.lastrowid


def log_feedback(
    conn: sqlite3.Connection,
    call_id: int,
    reaction: str,
    comment: Optional[str] = None,
) -> None:
    """Record a user reaction (e.g. thumbs up/down) to a previously logged LLM call."""
    conn.execute(
        "INSERT INTO feedback (call_id, reaction, comment) VALUES (?, ?, ?)",
        (call_id, reaction, comment),
    )


def load_llm_calls(conn: sqlite3.Connection, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load logged LLM call rows, optionally filtered to one batch. Defaults to all batches."""
    columns = [
        "id", "created_at", "batch_id", "approach", "query", "predicted_answer",
        "answered", "answer", "latency_seconds", "model", "prompt_tokens",
        "completion_tokens", "total_tokens", "cost_usd",
    ]
    if batch_id is None:
        cur = conn.execute(f"SELECT {', '.join(columns)} FROM llm_calls ORDER BY id")
    else:
        cur = conn.execute(
            f"SELECT {', '.join(columns)} FROM llm_calls WHERE batch_id = ? ORDER BY id",
            (batch_id,),
        )
    return [dict(zip(columns, row)) for row in cur.fetchall()]

