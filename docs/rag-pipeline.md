# RAG Retrieval Pipeline

This document describes the retrieval pipeline used by `RAGService` (`app/services/rag_service.py`), implemented in `app/services/retrieval_service.py` and its collaborators. It replaces the earlier single-pass "embed question → dense search → generate" flow with a three-stage pipeline: **pre-retrieval optimization**, **hybrid search**, and **post-retrieval reranking**.

## Why

Dense-only retrieval (cosine similarity over OpenAI embeddings) has two systematic weaknesses:

- **Short/vague queries retrieve poorly** — a raw user question often doesn't embed close to the answer passage it's asking about.
- **Exact terms get blurred** — embeddings are bad at exact keyword, ID, or acronym matching (e.g. an invoice number, a product code) that a plain keyword search would nail immediately.

And a single retrieval pass has no way to correct for either system's individual ranking mistakes before those chunks get handed to the LLM.

## Pipeline

```
question
  │
  ▼
┌─────────────────────────┐
│ 1. Pre-retrieval         │
│  - Query rewrite (LLM)   │  → rewritten_query
│  - HyDE (LLM)             │  → hyde_passage
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 2. Hybrid search          │
│  - Dense: Pinecone search  │ (embed(hyde_passage), top candidate_k)
│  - Lexical: BM25 search     │ (rewritten_query, top candidate_k)
│  - Fuse via RRF            │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 3. Post-retrieval         │
│  - Dedupe (filename+idx)  │
│  - Cross-encoder rerank    │ → final top_k chunks
└─────────────────────────┘
  │
  ▼
RAGService._build_context() / _format_sources() (unchanged)
```

All of this is orchestrated by `RetrievalService.retrieve(question, top_k, namespace)` (`app/services/retrieval_service.py`), called from `RAGService.generate_answer()` and `RAGService.get_similar_chunks()` in place of the old direct `EmbeddingService` + `VectorService.search()` call. It returns chunks in the exact same shape `VectorService.search()` always produced (`id`, `score`, `text`, `metadata`), so nothing downstream needed to change.

### 1. Pre-retrieval: query rewrite + HyDE

Two LLM calls (`gpt-4o-mini`, temperature 0), each with a safe fallback to the original question on failure:

- **Query rewrite** (`_rewrite_query`): turns the raw question into a clean, standalone search query — fixes typos, expands obvious acronyms, strips filler. Used as the query text for the BM25 leg.
- **HyDE** (`_generate_hyde_passage`, "Hypothetical Document Embeddings"): asks the LLM to write a short passage that *would* answer the question, as if pulled from a reference document, then embeds that passage instead of the raw question for the dense search leg. A generated answer-shaped passage embeds much closer to real answer chunks than a short interrogative question does.

Both are individually toggleable via config (`ENABLE_QUERY_REWRITE`, `ENABLE_HYDE`) in case either step's latency/cost isn't worth it for a given deployment.

### 2. Hybrid search: dense + BM25, fused with RRF

- **Dense leg**: `VectorService.search()` (Pinecone, cosine similarity) using the HyDE embedding, fetching `top_k * RERANK_CANDIDATE_MULTIPLIER` candidates.
- **Lexical leg**: `BM25Service.search()` (`app/services/bm25_service.py`), a `rank_bm25.BM25Okapi` index built in-memory from `CacheService` chunk data, using the rewritten query. Fetches the same candidate count.
- **Fusion**: Reciprocal Rank Fusion (`RetrievalService._reciprocal_rank_fusion`) — each chunk's fused score is `Σ 1/(RRF_K + rank + 1)` across whichever ranked list(s) it appears in. RRF only looks at rank position, not raw score magnitude, which sidesteps the standard hybrid-search failure mode of trying to normalize BM25 scores against cosine similarities (they're on incompatible scales).

**BM25 corpus source**: built from `CacheService.load_chunks_and_embeddings()` (full chunk text), *not* from Pinecone metadata — Pinecone only stores the first 1000 characters of each chunk's text (`VectorService.add_documents`), which would silently truncate longer chunks for lexical matching.

**Known limitation**: neither `CacheService` nor `StorageBackend` track Pinecone `namespace` — the BM25 corpus spans *all* cached documents, not just the ones in the Pinecone namespace being searched. This is fine while the app is effectively single-namespace (`"default"`), but will need `StorageBackend`/`CacheService` to start tracking namespace if multi-tenant namespaces become real.

### 3. Post-retrieval: dedupe + rerank

- **Dedupe** (`RetrievalService._dedupe`): the same chunk can appear in both the dense and lexical result lists (RRF already merges by id, but this is a second guard keyed on `(filename, chunk_index)` in case of id-format edge cases).
- **Rerank** (`RerankerService.rerank`, `app/services/reranker_service.py`): a local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`) scores the *original* question against each candidate chunk directly (not via embeddings), which is more accurate than either retrieval leg alone, then truncates to the final `top_k`. The model is lazy-loaded on first use so importing the service doesn't pay the model-download cost.

## Config knobs (`app/config.py`)

| Setting | Default | Controls |
|---|---|---|
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Which sentence-transformers cross-encoder model reranks chunks |
| `RERANK_CANDIDATE_MULTIPLIER` | `4` | How many candidates (`top_k * this`) are fetched per retrieval leg before rerank truncates to `top_k` |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant (standard default; larger = flatter weighting across ranks) |
| `ENABLE_QUERY_REWRITE` | `True` | Toggle the pre-retrieval LLM query-rewrite step |
| `ENABLE_HYDE` | `True` | Toggle the pre-retrieval HyDE passage-generation step |

## New dependencies

- `rank-bm25` — pure-Python `BM25Okapi`, no external service.
- `sentence-transformers` — local cross-encoder inference, no external API/key.

## Testing

- `tests/units/test_bm25_service.py` — BM25 ranking correctness against a fake `CacheService`.
- `tests/units/test_reranker_service.py` — rerank ordering/truncation against a fake cross-encoder (real model never loaded in tests).
- `tests/units/test_retrieval_service.py` — RRF fusion math, dedupe, and full `retrieve()` flow against fakes for every collaborator (pre-retrieval LLM calls disabled via settings to keep tests offline).
