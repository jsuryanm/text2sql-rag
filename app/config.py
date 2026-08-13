"""
Configuration management using Pydantic Settings.
Loads environment variables from .env file.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Multi-Source RAG + Text-to-SQL"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    ROOT_PATH: str = ""  # Set to "/prod" for API Gateway, empty for local development

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None  # Required for embeddings and RAG

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None  # Required for vector storage
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "rag-documents"

    # Supabase/PostgreSQL Configuration
    DATABASE_URL: Optional[str] = None  # Required for Text-to-SQL

    # OPIK Monitoring
    OPIK_API_KEY: Optional[str] = None  # Optional for monitoring
    OPIK_PROJECT_NAME: str = "text2sql"  # Add this line with your custom project name

    # Vanna 2.0 Configuration (Text-to-SQL)
    SQL_LLM_MODEL: str = "gpt-4o-mini"  # OpenAI model for SQL generation
    SQL_EXAMPLES_NAMESPACE: str = "sql-examples"  # Namespace within Pinecone index

    # SQL LLM Configuration for Determinism
    SQL_LLM_TEMPERATURE: float = 0.0  # 0.0 = fully deterministic, 1.0 = creative (range: 0.0-2.0)
    SQL_LLM_TOP_P: float = 0.1  # Nucleus sampling threshold (range: 0.0-1.0)
    SQL_LLM_SEED: int = 42  # Random seed for reproducibility
    SQL_LLM_MAX_TOKENS: int = 2000  # Maximum tokens for SQL generation

    # Text Chunking Configuration
    CHUNK_SIZE: int = 512
    MIN_CHUNK_SIZE: int = 256  # Minimum chunk size - smaller chunks will be merged
    CHUNK_OVERLAP: int = 50

    # Storage Backend Configuration
    STORAGE_BACKEND: str = "local"  # Options: "local", "s3"

    # Storage paths (auto-detects Lambda environment)
    @property
    def UPLOAD_DIR(self) -> str:
        # Use /tmp in Lambda/production, data/ locally
        if self.ENVIRONMENT == "production" or self.STORAGE_BACKEND == "s3":
            return "/tmp/uploads"
        return "data/uploads"

    @property
    def CACHE_DIR(self) -> str:
        # Use /tmp in Lambda/production, data/ locally
        if self.ENVIRONMENT == "production" or self.STORAGE_BACKEND == "s3":
            return "/tmp/cached_chunks"
        return "data/cached_chunks"

    # S3 Storage Configuration (for Lambda deployment)
    S3_CACHE_BUCKET: str = "rag-cache-bucket"
    AWS_REGION: str = "us-east-1"
    # AWS credentials from environment or IAM role (recommended for Lambda)
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are read automatically by boto3

    # Upstash Redis Configuration (Query-level caching)
    UPSTASH_REDIS_URL: Optional[str] = None  # Optional - app works without caching
    UPSTASH_REDIS_TOKEN: Optional[str] = None  # Optional - app works without caching

    # Cache TTL Configuration (in seconds)
    CACHE_TTL_EMBEDDINGS: int = 604800  # 7 days - embeddings are static
    CACHE_TTL_RAG: int = 3600           # 1 hour - may change with new documents
    CACHE_TTL_SQL_GEN: int = 86400      # 24 hours - schema relatively stable
    CACHE_TTL_SQL_RESULT: int = 900     # 15 minutes - data changes frequently

    model_config = SettingsConfigDict(env_file='.env',
                                      case_sensitive=False,
                                      env_file_encoding='utf-8',
                                      extra='ignore')

    @property
    def is_lambda(self) -> bool:
        """Check if running in AWS Lambda environment."""
        return os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None


# Global settings instance
settings = Settings()
