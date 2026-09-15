# Multi-Source RAG + Text-to-SQL

A FastAPI-based platform for hybrid document retrieval (RAG) and natural-language-to-SQL querying. Combines dense (Pinecone) and lexical (BM25) search with cross-encoder reranking for document Q&A, alongside Vanna-based text-to-SQL against Postgres/Supabase.

> **Project status:** in active development. No FastAPI app is wired up yet (`app/main.py` is an empty stub), so the API does not currently run. Most core services (storage, caching, embeddings, retrieval) are implemented and tested — see [Project State](#project-state) below.

## Features

- **Document ingestion** via [Docling](https://github.com/DS4SD/docling) (layout-aware parsing, heading/page/caption preservation) with an `unstructured`-based fallback path, and token-aware chunking.
- **Hybrid retrieval pipeline** (`RetrievalService`): LLM query rewriting + HyDE, dense search via Pinecone fused with BM25 lexical search using Reciprocal Rank Fusion, then cross-encoder reranking. See [docs/rag-pipeline.md](docs/rag-pipeline.md) for full details.
- **Two-tier caching**: document-level cache (chunks/embeddings/metadata) backed by a pluggable local-disk or S3 storage backend, plus a query/embedding-level cache backed by Upstash Redis (runs safely in pass-through mode when Redis isn't configured).
- **Text-to-SQL** against Postgres/Supabase via Vanna (planned — `sql_service.py` is not yet implemented).
- **Deployable to AWS Lambda** via Mangum, with Dockerfiles for both standard and Tesseract-enabled OCR images.

## Tech Stack

- **API**: FastAPI, Mangum (Lambda)
- **Document parsing**: Docling, `unstructured`, `semchunk`
- **Embeddings & vector store**: OpenAI embeddings, Pinecone (`langchain-pinecone`)
- **Lexical search & reranking**: `rank-bm25`, `sentence-transformers` cross-encoder
- **Text-to-SQL**: LangChain, SQLAlchemy/psycopg against Postgres/Supabase
- **Orchestration**: LangGraph
- **Caching**: Upstash Redis (query-level), local disk or S3 (document-level)
- **Monitoring**: Opik
- **Dependency management**: [uv](https://github.com/astral-sh/uv)

## Project State

| Area                                                                          | Status                  |
| ----------------------------------------------------------------------------- | ----------------------- |
| `app/main.py` (FastAPI app)                                                 | Not implemented (empty) |
| `evaluate.py`, `lambda_handler.py`, `sql_service.py`                    | Not implemented (empty) |
| `app/config.py`, `app/logging_config.py`, `app/utils.py`                | Implemented             |
| Storage backends (local, S3),`CacheService`, `QueryCacheService`          | Implemented             |
| `EmbeddingService`, `DocumentService`, `DoclingService`                 | Implemented             |
| `VectorService` (Pinecone), `RAGService`                                  | Implemented             |
| `RetrievalService`, `BM25Service`, `RerankerService` (hybrid retrieval) | Implemented             |

Since there's no wired-up FastAPI app yet, there are no integration tests — only unit tests against the implemented services.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management
- API keys/credentials as needed: OpenAI, Pinecone, Upstash Redis (optional), Supabase/Postgres (for text-to-SQL), Opik (optional)

### Installation

```bash
# Install core dependencies
uv sync

# Install with dev, test, eval, and AWS extras
uv sync --extra all
```

### Configuration

Copy your environment variables into a `.env` file at the project root. Key settings (see `app/config.py` for the full list):

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=rag-documents
DATABASE_URL=
UPSTASH_REDIS_URL=
UPSTASH_REDIS_TOKEN=
STORAGE_BACKEND=local   # or "s3"
ENVIRONMENT=development
```

`STORAGE_BACKEND` and `ENVIRONMENT` together control whether document/cache storage lives under `data/` locally or `/tmp/` (required for Lambda's read-only filesystem).

### Running Tests

```bash
# Full suite (coverage runs automatically, HTML report to htmlcov/)
uv run pytest

# A single test
uv run pytest tests/units/test_cache_service.py::TestClearCache::test_clear_all_documents

# Skip slow tests
uv run pytest -m "not slow"
```

### Linting, Formatting, Type Checking

```bash
uv run ruff check .
uv run black .
uv run mypy app
```

## Architecture

- **Config**: a single `pydantic-settings` object (`app.config.settings`) loaded from `.env`, with per-purpose cache TTLs and environment-aware storage path resolution.
- **Storage abstraction**: an abstract `StorageBackend` interface implemented by both local-disk and S3 backends, so the same `CacheService` API works in local dev and in Lambda. Documents are content-addressed by a SHA-256 hash for deduplication.
- **Retrieval pipeline**: three stages — pre-retrieval (query rewrite + HyDE), hybrid search (dense Pinecone + lexical BM25 fused via Reciprocal Rank Fusion), and post-retrieval (dedupe + cross-encoder reranking). Full write-up in [docs/rag-pipeline.md](docs/rag-pipeline.md).
- **Two parallel document-parsing paths** exist (`document_service.py` for fast/native parsing, `docling_service.py` for layout-aware parsing) unified behind `document_service.parse_and_chunk_with_context()`, which is the intended entry point for ingestion.

See [CLAUDE.md](CLAUDE.md) for the full architecture and implementation notes.

## Documentation

- [docs/rag-pipeline.md](docs/rag-pipeline.md) — hybrid retrieval pipeline design
- [docs/pytest-learning-guide.md](docs/pytest-learning-guide.md) — pytest tutorial written against this repo's test suite

## License

MIT
