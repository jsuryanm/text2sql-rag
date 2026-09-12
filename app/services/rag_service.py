from typing import List, Dict, Any, Optional 
import json 
import logging 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings 
from app.services.vector_service import VectorService
from app.services.embeddings_service import EmbeddingService
from app.services.query_cache_service import QueryCacheService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, api_key: str | None = None, query_cache_service = None):
        """
        Initialize the RAG service.

        Args:
            api_key: OpenAI API key (optional, uses settings if not provided)
            query_cache_service: Optional QueryCacheService for response caching
        """
        self.api_key = api_key or settings.OPENAI_API_KEY 
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required. Set the OPENAI_API_KEY in the .env file.")

        self.embedding_service = EmbeddingService(
            api_key=self.api_key,
            query_cache_service=query_cache_service
        )

        self.vector_service = VectorService(openai_api_key=self.api_key)
        self.query_cache_service : QueryCacheService = query_cache_service

        self.model = "gpt-4o-mini"
        self.temperature = 0.1 
        self.max_tokens = 1000 

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            max_token=self.max_tokens
        )

        # LCEL chain: prompt -> chat model 

        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that answers questions based on provided context. "
                "If the context doesn't contain enough information to answer the question, "
                "say so explicitly. Always base your answers on the provided context."
            ),
            ("human","{prompt}")
        ])

        self.chain = self.prompt_template | self.llm

    async def generate_answer(
        self,
        question: str,
        top_k: int = 3,
        namespace: str = "default",
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve relevant chunks and generate an answer.

        NEW: Implements query-level caching to save ~$0.05 per cache hit.
        - Cache key: hash(question + top_k)
        - Cache TTL: 1 hour (configurable)
        - Falls back to uncached if Redis unavailable

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default: 3)
            namespace: Pinecone namespace to search (default: "default")
            include_sources: Whether to include source citations (default: True)

        Returns:
            Dictionary containing:
                - question: The original question
                - answer: Generated answer from LLM
                - sources: List of source chunks used (if include_sources=True)
                - chunks_used: Number of chunks retrieved
                - model: LLM model used
                - cache_hit: Whether result came from cache (NEW)
                - cost_saved: Estimated cost saved if cache hit (NEW)
        """
        try:
            if self.query_cache_service and self.query_cache_service.enabled:
                cache_key = self.query_cache_service.get_rag_key(question=question, top_k=top_k)
                cached_result = self.query_cache_service.get(cache_key, cache_type='rag')

                if cached_result:
                    logger.info(f"RAG cache HIT for question: {question[:50]}...")
                    return {
                        **cached_result,
                        "cache_hit": True,
                        "cost_saved": "$0.60" # approximate cost per query
                    }

                # Step 1: Generate query embedding with usage tracking
                embeddings, embedding_usage = await self.embedding_service.generate_embeddings(texts=[question])
                query_embedding = embeddings[0]

                search_results = await self.vector_service.search(
                    
                )


        
        