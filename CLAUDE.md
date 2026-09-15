# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

FastAPI project ("Multi-Source RAG + Text-to-SQL"). No FastAPI app is wired up yet — `app/main.py` is empty, so `fastapi dev` / uvicorn won't run until it has an app object. The root `main.py` is an unrelated `uv init` placeholder ("Hello from multidata-rag!"), not the app entrypoint.

- Empty/unimplemented: `app/main.py`, `evaluate.py`, `lambda_handler.py`, `sql_service.py`
- Implemented: `app/config.py`, `app/logging_config.py`, `app/utils.py`, and all of `app/services/storage_backend.py`, `local_storage.py`, `s3_storage.py`, `cache_service.py`, `query_cache_service.py`, `embeddings_service.py`, `document_service.py`, `docling_service.py`, `vector_service.py`, `router_service.py`, `rag_service.py`, `retrieval_service.py`, `bm25_service.py`, `reranker_service.py`

Check line count before assuming a service works — the empty ones above are true 0-byte stubs.

## Commands

Managed with `uv` (see `uv.lock`).

- Install deps: `uv sync` (add `--extra all` for dev+test+eval+aws extras)
- Run tests: `uv run pytest`
- Run a single test: `uv run pytest tests/units/test_cache_service.py::TestClearCache::test_clear_all_documents`
- Skip slow tests: `uv run pytest -m "not slow"` (markers: `slow`, `integration`, `unit`, `aws`)
- Lint: `uv run ruff check .`
- Format: `uv run black .`
- Type check: `uv run mypy app`

Coverage runs automatically with pytest (`--cov=app`, HTML report to `htmlcov/`).

## Architecture

- **Intended stack** (from dependencies): FastAPI API; document ingestion via `docling`/`unstructured` + `semchunk`; OpenAI embeddings stored in Pinecone (`langchain-pinecone`); Vanna for text-to-SQL against Postgres/Supabase (SQLAlchemy/psycopg); LangGraph for orchestration; Upstash Redis for query-level caching; Opik for LLM monitoring; deployable to AWS Lambda via `mangum` (`Dockerfile.lambda`, `Dockerfile.lambda.with-tesseract`).

- **Config** (`app/config.py`): single `pydantic-settings` object `app.config.settings`, loads `.env`. `STORAGE_BACKEND` env var (`"local"` | `"s3"`) drives `UPLOAD_DIR`/`CACHE_DIR` properties — auto-switches to `/tmp/...` when `ENVIRONMENT=production` or backend is `s3`, since Lambda's filesystem is read-only outside `/tmp`. Per-purpose cache TTLs (embeddings/RAG/SQL-gen/SQL-result) are predefined for use by the caching services.

- **Storage abstraction** (`app/services/storage_backend.py`): abstract `StorageBackend` (exists/save_document/save_chunks/save_embeddings/save_metadata/load_*/delete/list_documents/get_stats) lets local dev and S3/Lambda share one interface, selected via `settings.STORAGE_BACKEND`. Both `local_storage.py` and `s3_storage.py` are fully implemented. Each document is stored under `{cache_dir}/{document_id}/` (local) or `{bucket}/{file_extension}/{document_id}/` (S3) with 4 files: `document.{ext}`, `chunks.json`, `embeddings.npy`, `metadata.json`. `document_id` is a SHA-256 hash of content, used as the dedup/cache key. `CacheService` (`cache_service.py`) sits on top of a `StorageBackend` and is the entry point application code should use — it auto-selects local vs. S3 from `settings.STORAGE_BACKEND` (with fallback to local if S3 init fails), and accepts an injected backend for testing.

- **Two-tier caching**: `CacheService` (document-level: chunks/embeddings/metadata, backed by `StorageBackend`) is distinct from `QueryCacheService` (`query_cache_service.py`, query/embedding-level, backed by Upstash Redis). `QueryCacheService` runs in a safe pass-through "disabled" mode when Redis credentials aren't configured — every `get()` is a miss and `set()`/`delete()` are no-ops, so the app keeps working without caching. `EmbeddingService` (`embeddings_service.py`) uses `QueryCacheService` internally to cache individual text embeddings for 7 days (`CACHE_TTL_EMBEDDINGS`) before calling OpenAI.

- **Two parallel document-parsing paths**, both implemented — decide which is canonical before building ingestion on top of either:
  - `document_service.parse_document()` + `chunk_text()`: fast native read for `.txt`/`.csv`/`.log`/`.json`, falls back to `unstructured.partition.auto` (strategy `"fast"`, no OCR) for everything else (pdf/docx). Token-based chunking via `langchain_text_splitters.TokenTextSplitter`.
  - `docling_service.convert_document()` + `chunk_with_hybrid()`: layout-aware path using Docling + `HybridChunker`, gated by a `DOCLING_AVAILABLE` import flag (falls back to `document_service` via `fallback_to_unstructured()` when Docling isn't installed). Preserves heading hierarchy, page numbers, and captions per chunk.
  - `document_service.parse_and_chunk_with_context()` is the unified entry point: routes plain-text formats to the fast path directly, everything else through Docling with a fallback to token-based chunking on `ImportError`/failure — callers should generally use this rather than calling either path directly.

- **Validation** (`app/utils.py`): `FileValidator` (extension allowlist pdf/docx/doc/csv/json/txt, 50MB max) and `QueryValidator` (question length bounds, dangerous-SQL regex guard) plus `ErrorResponse` helpers for consistent FastAPI error bodies — reuse these instead of re-validating ad hoc.

- **Logging** (`app/logging_config.py`): app-wide logging setup. Services use `logging.getLogger("rag_app.<service>")` or `__name__` — match this convention in new modules.

- **Retrieval pipeline** (`app/services/retrieval_service.py`, `bm25_service.py`, `reranker_service.py`): `RAGService.generate_answer()`/`get_similar_chunks()` delegate retrieval to `RetrievalService.retrieve()` rather than calling `VectorService.search()` directly. Three stages: (1) pre-retrieval — LLM query rewrite + HyDE (hypothetical document embeddings), each toggleable via `settings.ENABLE_QUERY_REWRITE`/`ENABLE_HYDE`; (2) hybrid search — dense (Pinecone via `VectorService`) + lexical (`BM25Service`, an in-memory `rank_bm25.BM25Okapi` index built from `CacheService` chunk text, not Pinecone's 1000-char-truncated metadata) fused via Reciprocal Rank Fusion; (3) post-retrieval — dedupe + `RerankerService` (local `sentence-transformers` cross-encoder, lazy-loaded). New deps: `rank-bm25`, `sentence-transformers`. Known limitation: `BM25Service`'s corpus is not namespace-scoped (neither `CacheService` nor `StorageBackend` track Pinecone namespace) — fine while the app is single-namespace (`"default"`). Full details in `docs/rag-pipeline.md`.

## Testing

Tests live in `tests/units/`, covering every implemented service (storage backends, `CacheService`, `QueryCacheService`, `EmbeddingService`, `document_service`, `docling_service`, `vector_service`, `bm25_service`, `reranker_service`, `retrieval_service`). No FastAPI/integration tests yet since no app is wired up.

Shared fixtures (`sample_chunks`, `sample_embeddings`, `sample_metadata`, `temp_document`) live in `tests/conftest.py` and are auto-available to every test file — don't redefine them locally. External dependencies are never hit directly in tests:
- S3 (`s3_storage.py`) is faked with `moto`'s `mock_aws()`.
- Redis (`query_cache_service.py`) and OpenAI (`embeddings_service.py`) are faked with small hand-written fake client classes rather than `unittest.mock.MagicMock`, injected via constructor args or attribute assignment.
- Docling (`docling_service.py`) is faked via `monkeypatch.setattr` on its `DocumentConverter`/`HybridChunker` classes, since the real library is a heavy optional dependency.
- The reranker's `sentence-transformers` `CrossEncoder` (`reranker_service.py`) is faked the same way — real model weights are never downloaded in tests.

`asyncio_mode = "auto"` is set in `pyproject.toml`, so `async def test_...` functions run automatically without needing `@pytest.mark.asyncio`.

`docs/pytest-learning-guide.md` is a from-scratch pytest tutorial written against this repo's actual test suite (fixtures, mocking techniques, async tests, markers, coverage) — point beginners there instead of re-explaining pytest basics.
