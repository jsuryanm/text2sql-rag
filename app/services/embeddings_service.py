from typing import List, Dict, Optional, Tuple

import tiktoken
import logging 

from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.services.query_cache_service import QueryCacheService

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings with OpenAI text-embedding-3-small"""

    def __init__(self, api_key: str | None = None, query_cache_service: QueryCacheService = None):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            query_cache_service: Optional QueryCacheService for embedding caching
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file")

        self.model = "text-embedding-3-small"
        self.dimensions = 1536
        self.client = OpenAIEmbeddings(model=self.model, api_key=self.api_key)

        self._tokenizer = tiktoken.get_encoding('o200k_base')

        if query_cache_service is not None:
            self.query_cache_service = QueryCacheService() 

    def _count_tokens(self, texts: List[str]) -> int:
        """Count tokens across a list of texts using the model's tokenizer."""
        return sum(len(self._tokenizer.encode(text)) for text in texts)

    async def generate_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Optional[Dict]]:
        """
        Generate embeddings for a list of texts with caching support.

        NEW: Implements per-text caching to avoid re-computing identical embeddings.
        - Cache key: hash(text)
        - Cache TTL: 7 days (embeddings are deterministic)
        - Falls back to uncached if Redis unavailable

        Args:
            texts: List of text strings to embed

        Returns:
            Tuple of (embeddings, usage_info) where:
            - embeddings: List of embedding vectors (each is a list of floats)
            - usage_info: Dict with token counts and model info for cost tracking

        Raises:
            Exception: If embedding generation fails
        """
        if not texts:
            return [], None 

        if self.query_cache_service and self.query_cache_service.enabled:
            embeddings = []
            texts_to_generate = []
            text_indices = []  # Track original indices for uncached texts
            cache_hits = 0 
            cache_misses = 0 

            for i, text in enumerate(texts):
                cached_key = self.query_cache_service.get_embeddings_key(text)
                cached = self.query_cache_service.get(cached_key, cache_type='embedding')

                if cached and "embedding" in cached:
                    embeddings.append(cached["embedding"])
                    cache_hits += 1 

                else:
                    embeddings.append(None)
                    texts_to_generate.append(text)
                    text_indices.append(i)
                    cache_misses += 1 

            # Generate embeddings for uncached texts
            if texts_to_generate:

                try:
                    new_embeddings = await self.client.aembed_documents(texts_to_generate)

                    # Cache new embeddings and fill in results
                    for idx, embedding in zip(text_indices, new_embeddings):
                        embeddings[idx] = embedding

                        # Cache individual embedding
                        cache_key = self.query_cache_service.get_embeddings_key(texts[idx])
                        cache_value = {
                            "embedding": embedding,
                            "model": self.model,
                            "text_length":len(texts[idx])
                        }

                        ttl = settings.CACHE_TTL_EMBEDDINGS
                        self.query_cache_service.set(
                            cache_key,
                            cache_value,
                            ttl=ttl,
                            cache_type='embedding'
                        )

                        logger.debug(f"Embedding cache: {cache_hits} hits, {cache_misses} misses"
                                     f" ({cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate)")

                        token_count = self._count_tokens(texts_to_generate)
                        usage_info = {
                            "prompt_tokens": token_count,
                            "total_counts": token_count,
                            "model": self.model,
                            "cache_hits": cache_hits,
                            "cache_misses": cache_misses
                        }

                        return embeddings, usage_info

                except Exception as e:
                    raise Exception(f"Failed to generate embeddings: {str(e)}")

            else:
                # All embeddings came from cache
                logger.debug(f"Embedding cache: {cache_hits} hits, 0 misses (100% cache hit rate)")

                return embeddings, {
                    "cache_hits": cache_hits,
                    "cache_misses": 0, 
                    "model": self.model
                }
            
        # No cache available - generate all embeddings
        try:
            embeddings = await self.client.aembed_documents(texts)

            token_count = self._count_tokens(texts)
            usage_info = {
                "prompt_tokens": token_count,
                "total_count": token_count,
                "model": self.model
            }

            return embeddings, usage_info 

        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")