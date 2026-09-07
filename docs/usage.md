# Usage

## Running the app

Make an ```.env``` file with `OPENAI_API_KEY`
(it's used as the default LLM client). `NCBI_API_KEY` is
optional, only used for the live PubMed lookup in retrieval, which is not used
by default.

```bash
docker compose up --build
```

This builds the image, runs the `ingest` service once (populates `data/app.db`
and the FAISS vector stores if anything in `knowledge_base/` changed, skips it
otherwise), then starts:

- **API** - http://localhost:8000
- **Dashboard** — http://localhost:8501

`data/` can contain only the ```test_queries.json``` file (e.g. a fresh clone) — `ingest` always does a full
`data_prep` run the first time, since it has no prior fingerprint to compare
against. `knowledge_base/` must already contain the source PDFs though;
`ingest` reads from there, not from `data/`.

On first run, `api` needs a couple of minutes to download and load the
embedding models before it's ready — `dashboard` waits for `api`'s healthcheck
to pass before starting, so you shouldn't hit a connection-refused error from
the dashboard; just give it a moment. Downloaded models are cached under
`data/hf_cache/`, so subsequent runs start faster.

To stop the app, `Ctrl+C` then `docker compose down` (data in `data/`
persists on disk). If you add/remove/change a PDF in
`knowledge_base/`, just re-run `docker compose up --build` — `ingest` detects
the change and reprocesses automatically.

## Retrieval Evaluation

To run the retrieval evaluation (retrieves the chunks for the test queries with
every method, then reports hit rate/MRR against ground truth (```test_queries.json```) and overlap and some summary statistics of the methods):

```bash
docker compose run --build --rm eval-retrieval
```

## LLM Evaluation

To run the LLM evaluation batch (not part of the default `up`), where two GPT models and
three retrieval methods are used (together 6 different combinations) and evaluated against
ground truth (```test_queries.json```) to obtain the combination with the highest accuracy.

```bash
docker compose run --build --rm eval
```




