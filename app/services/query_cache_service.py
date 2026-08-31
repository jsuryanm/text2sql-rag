import hashlib 
import json 
import logging 
from pathlib import Path 
from typing import List, Dict, Any, Optional 
import numpy as np 

logger = logging.getLogger(__name__)

class QueryCacheService:
    """Redis-based cache service for query results, embeddings, and SQL."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_token: Optional[str] = None
    ):
        """
        Initialize Upstash Redis connection.

        Args:
            redis_url: Upstash Redis REST URL
            redis_token: Upstash Redis REST token

        Note: If credentials not provided, service operates in pass-through mode
        (all operations return cache miss, but app continues working).
        """
        self.enabled = False 
        self.client = None 

        self.stats = {
            "embeddings": {"hits": 0, "misses": 0},
            "rag": {"hits": 0, "misses": 0},
            "sql_gen": {"hits": 0, "misses": 0},
            "sql_result": {"hits": 0, "misses": 0},
        }

        if redis_url and redis_token:
            try:
                from upstash_redis import Redis 

                self.client = Redis(url=redis_url,
                                    token=redis_token)

                self.client.ping()
                # tests if connection to Redis server is alive and functional

                self.enabled = True
                logger.info(f"Upstash Redis is connected")

            except ImportError:
                logger.warning(f"upstash-redis is not installed. Cache disabled. "
                               f"Install with pip install upstash-redis")

            except Exception as e:
                logger.warning(f" Failed to connect to Upstash Redis: {e} "
                               f"Cache disabled. App will continue without caching")
        
        else:
            logger.info(f"Upstash Redis credentials not configured. Cache disabled "
                        f"set the UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN to enable.")

    def _compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash of text for cache keys."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON string for storage."""
        return json.dumps(value, default=str)

    def _deserialize(self, value: str) -> Any:
        """Deserialize JSON string to Python object."""
        return json.loads(value)

    def get(self):
        pass 

    # Cache Key Generators
    def get_embeddings_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        text_hash = self._compute_hash(text)
        return f"embeddings:{text_hash}"

    def get_rag_key(self, question: str, top_k: int) -> str:
        """Generate cache key for RAG response."""
        question_hash = self._compute_hash(question.lower())
        return f"rag:{question_hash}:{top_k}"

    def get_sql_gen_key(self, question: str) -> str:
        """Generate cache key for SQL generation."""
        question_hash = self._compute_hash(question.lower())
        return f"sql_gen:{question_hash}"

    def get_sql_result_key(self, sql_query: str) -> str:
        """
        Generate cache key for SQL result.

        Normalizes SQL (removes extra whitespace, lowercase) for better cache hits.
        """
        normalized_sql = " ".join(sql_query.strip().lower().split())
        sql_hash = self._compute_hash(normalized_sql)
        return f"sql_result:{sql_hash}"

    # Statistics Tracking
    
    def _record_hit(self, cache_type: str):
        """Record cache hit for statistics."""
        if cache_type in self.stats:
            self.stats[cache_type]["hits"] += 1 

    def _record_miss(self, cache_type: str):
        """Record cache miss for statistics."""
        if cache_type in self.stats:
            self.stats[cache_type]["misses"] += 1

    def get_stats(self) -> Dict:
        """
        Get cache hit/miss statistics.

        Returns:
            Dictionary with hit rates for each cache type
        """
        stats_with_rates = {}

        for cache_type, counts in self.stats.items():
            total = counts['hits'] + counts['misses']
            hit_rate = (counts["hits"] / total * 100) if total > 0 else 0 
            stats_with_rates[cache_type] = {
                "hits": counts["hits"],
                "misses": counts["misses"],
                "total_queries": total,
                "hit_rate": f"{hit_rate:.1f}%",
            }

        return {
            "enabled": self.enabled,
            "cache_types": stats_with_rates
        }

    def reset_stats(self):
        """Resets statistics counters"""
        for cache_type in self.stats:
            self.stats[cache_type] = {'hits': 0, 'misses': 0}

        logger.info("Cache statistics reset")
       
    # health check
    def health_check(self) -> Dict:
        """
        Check Redis connection health.

        Returns:
            Status dictionary
        """
        if not self.enabled:
            return {"status": "disabled", "message": "Redis cache not configured"}

        try:
            self.client.ping()
            return {"status": "healthy", "message": "Redis connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Redis error: {str(e)}"}
