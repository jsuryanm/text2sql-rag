# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Early-stage FastAPI project ("Multi-Source RAG + Text-to-SQL"). Many `app/services/*.py` files are empty stubs — check line count before assuming a service works:

- Empty/unimplemented: `app/main.py` (no FastAPI app wired up yet), `evaluate.py`, `lambda_handler.py`, `app/services/rag_service.py`, `sql_service.py`, `vector_service.py`, `embeddings_service.py`, `router_service.py`, `cache_service.py`, `query_cache_service.py`, `s3_storage.py`
- Implemented: `app/config.py`, `app/logging_config.py`, `app/utils.py`, `app/services/storage_backend.py` (abstract interface), `app/services/local_storage.py` (concrete impl), `app/services/docling_service.py` (partial — `chunk_with_hybrid()` is `pass`), `app/services/document_service.py` (partial — `parse_document()` done, `chunk_text()` body incomplete)

No app entrypoint exists yet, so `fastapi dev` / uvicorn won't run until `app/main.py` has an app object.

## Commands

Managed with `uv` (see `uv.lock`).

- Install deps: `uv sync` (add `--extra all` for dev+test+eval+aws extras)
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/test_storage_backends.py::TestClass::test_name`
- Skip slow tests: `uv run pytest -m "not slow"` (markers: `slow`, `integration`, `unit`, `aws`)
- Lint: `uv run ruff check .`
- Format: `uv run black .`
- Type check: `uv run mypy app`

Coverage runs automatically with pytest (`--cov=app`, HTML report to `htmlcov/`).

## Architecture

- **Intended stack** (from dependencies): FastAPI API; document ingestion via `docling`/`unstructured` + `semchunk`; OpenAI embeddings stored in Pinecone (`langchain-pinecone`); Vanna for text-to-SQL against Postgres/Supabase (SQLAlchemy/psycopg); LangGraph for orchestration; Upstash Redis for query-level caching; Opik for LLM monitoring; deployable to AWS Lambda via `mangum` (`Dockerfile.lambda`, `Dockerfile.lambda.with-tesseract`).

- **Config** (`app/config.py`): single `pydantic-settings` object `app.config.settings`, loads `.env`. `STORAGE_BACKEND` env var (`"local"` | `"s3"`) drives `UPLOAD_DIR`/`CACHE_DIR` properties — auto-switches to `/tmp/...` when `ENVIRONMENT=production` or backend is `s3`, since Lambda's filesystem is read-only outside `/tmp`. Per-purpose cache TTLs (embeddings/RAG/SQL-gen/SQL-result) are predefined even though the caching services that would use them aren't implemented yet.

- **Storage abstraction** (`app/services/storage_backend.py`): abstract `StorageBackend` (exists/save_document/save_chunks/save_embeddings/save_metadata/load_*/delete/list_documents/get_stats) lets local dev and S3/Lambda share one interface, selected via `settings.STORAGE_BACKEND`. `local_storage.py` is the only concrete implementation currently (`s3_storage.py` is a stub) — stores each document under `{cache_dir}/{document_id}/` with 4 files: `document.{ext}`, `chunks.json`, `embeddings.npy`, `metadata.json`. `document_id` is a SHA-256 hash of content, used as the dedup/cache key.

- **Two parallel, unwired document-parsing paths** — decide which is canonical before building ingestion on top of either:
  - `document_service.parse_document()`: fast native read for `.txt`/`.csv`/`.log`/`.json`, falls back to `unstructured.partition.auto` for everything else (pdf/docx).
  - `docling_service.convert_document()`: layout-aware path using Docling + `HybridChunker`, gated by a `DOCLING_AVAILABLE` import flag. Only conversion is implemented; `chunk_with_hybrid()` is a no-op stub.

- **Validation** (`app/utils.py`): `FileValidator` (extension allowlist pdf/docx/doc/csv/json/txt, 50MB max) and `QueryValidator` (question length bounds, dangerous-SQL regex guard) plus `ErrorResponse` helpers for consistent FastAPI error bodies — reuse these instead of re-validating ad hoc.

- **Logging** (`app/logging_config.py`): app-wide logging setup. Services use `logging.getLogger("rag_app.<service>")` or `__name__` — match this convention in new modules.

## Testing

Only `tests/test_storage_backends.py` exists, covering the `StorageBackend` interface (local + S3 via `moto`). No FastAPI/integration tests yet since no app is wired up.
