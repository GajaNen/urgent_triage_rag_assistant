"""
Orchestration for knowledge-base ingestion, using dlt.

Why dlt over Airflow: this project has one linear batch job (detect new/changed
knowledge-base PDFs -> re-run DataPrep), not a multi-task DAG with branching or
cross-task dependencies. dlt gives a runnable, stateful ingestion pipeline in a
few lines and needs no extra long-running services (webserver, scheduler,
metadata Postgres) to containerize, unlike Airflow.

Run manually with `python src/ingest_pipeline.py`, or on a schedule (cron,
docker-compose healthcheck-triggered job, etc.).
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator

import dlt

from data_prep import DataPrep

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
PIPELINE_DIR = Path(__file__).parent.parent / "data" / "dlt_pipeline"
STATE_FILE = PIPELINE_DIR / "last_ingested.txt"


@dlt.resource(name="knowledge_base_files", write_disposition="merge", primary_key="filename")
def kb_files() -> Iterator[Dict[str, Any]]:
    """Yield one row per source PDF with its modification time, for ingestion history/observability."""
    for pdf_path in sorted(p for p in KB_DIR.glob("*.pdf") if not p.stem.endswith("_unredacted")):
        stat = pdf_path.stat()
        yield {
            "filename": pdf_path.name,
            "modified_at": stat.st_mtime,
            "size_bytes": stat.st_size,
        }


def _current_fingerprint() -> str:
    """Hash of (filename, mtime) for every knowledge-base PDF, to detect any change in the fileset."""
    pdfs = sorted(p for p in KB_DIR.glob("*.pdf") if not p.stem.endswith("_unredacted"))
    fingerprint = "|".join(f"{p.name}:{p.stat().st_mtime}" for p in pdfs)
    return hashlib.sha256(fingerprint.encode()).hexdigest()


def run_ingestion(force: bool = False) -> bool:
    """Record the current knowledge-base fileset via dlt, then re-run DataPrep only if it changed."""
    pipeline = dlt.pipeline(
        pipeline_name="kb_ingestion",
        destination=dlt.destinations.duckdb(str(PIPELINE_DIR / "state.duckdb")),
        dataset_name="kb_ingestion",
    )
    pipeline.run(kb_files())

    fingerprint = _current_fingerprint()
    previous = STATE_FILE.read_text().strip() if STATE_FILE.exists() else None
    changed = force or fingerprint != previous

    if changed:
        print("Detected new/changed knowledge base files, running DataPrep")
        DataPrep().run()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(fingerprint)
    else:
        print("No new/changed knowledge base files, skipping DataPrep")

    return changed


if __name__ == "__main__":
    run_ingestion()
