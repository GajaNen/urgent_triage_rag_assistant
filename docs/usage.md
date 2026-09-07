# Usage

## Running the app

From the repo root, with a `.env` file containing at least `OPENAI_API_KEY` (and
`NCBI_EMAIL`/`NCBI_API_KEY` if you use those):

```bash
docker compose up --build
```

This builds the image, runs the `ingest` service once (populates `data/app.db`
and the FAISS vector stores if anything in `knowledge_base/` changed, skips it
otherwise), then starts:

- **API** — http://localhost:8000 (interactive docs at `/docs`)
- **Dashboard** — http://localhost:8501

`data/` can be empty (e.g. a fresh clone) — `ingest` always does a full
`data_prep` run the first time, since it has no prior fingerprint to compare
against. `knowledge_base/` must already contain the source PDFs though;
`ingest` reads from there, not from `data/`.

To re-run the LLM evaluation batch on demand (not part of the default `up`):

```bash
docker compose run eval
```

To re-run the retrieval evaluation on demand (retrieves the test queries with
every method, then reports hit rate/MRR/overlap against ground truth):

```bash
docker compose run eval-retrieval
```

## Sharing with others

**Quick, temporary demo (public URL, no deployment)** — expose your local
ports with a tunnel while `docker-compose up` is running:

```bash
ngrok http 8501   # dashboard
ngrok http 8000   # api, if they need direct API access too
```

Cloudflare Tunnel is a free alternative if you'd rather not use ngrok's
signup/rate limits.

**Same local network only** — find your machine's LAN IP (`ipconfig` on
Windows, `ifconfig`/`ip a` on macOS/Linux) and share `http://<your-ip>:8501`.
Works out of the box since the compose services already bind `0.0.0.0`.

**Persistent hosting** — deploy the same `docker-compose.yml` (using `docker compose`) to a small cloud
VM (DigitalOcean, Hetzner, AWS Lightsail, etc.), open ports 8000/8501 in the
firewall, and share `http://<vm-ip>:8501`. Preferable for anything that needs
to stay reachable after you close your laptop.
