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
        