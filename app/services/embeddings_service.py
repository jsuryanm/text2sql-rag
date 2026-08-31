from typing import List, Dict, Optional, Tuple

import tiktoken
import logging 

from langchain_openai import OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings with OpenAI text-embedding-3-small"""

    def __init__(self, api_key: str | None = None, query_cache_service=None):
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
        self.query_cache_service = query_cache_service

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
            cache_hits = 0 
            cache_misses = 0 

            for i, text in enumerate(texts):
                cached_key = 
        