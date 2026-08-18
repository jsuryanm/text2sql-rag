import json 
import io 
import logging 
from pathlib import Path
from typing import Dict, List 
import numpy as np 
import boto3 
from botocore.exceptions import ClientError
from botocore.config import Config 
from app.services.storage_backend import StorageBackend
from app.config import settings 

logger = logging.getLogger(__name__)

"""
AWS S3 storage backend for Lambda deployment.
Note: Lambda has an empheral filesystem (contents are removed, once stop instance)

This stores documents in S3 organized by file type:
- s3://bucket/pdf/{hash}/document.pdf
- s3://bucket/pdf/{hash}/chunks.json
- s3://bucket/pdf/{hash}/embeddings.npy
- s3://bucket/pdf/{hash}/metadata.json

"""

class S3StorageBackend(StorageBackend):
    """
     Organizes documents by type in S3:
    - pdf/{doc_id}/document.pdf, chunks.json, embeddings.npy, metadata.json
    - txt/{doc_id}/document.txt, chunks.json, embeddings.npy, metadata.json
    - markdown/{doc_id}/document.md, chunks.json, embeddings.npy, metadata.json

    Each document gets 4 files: original document + cache files.
    """
    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or settings.S3_CACHE_BUCKET
        self.region = settings.AWS_REGION

        boto_config = Config(region_name=self.region,
                             retries={
                                 "max_attempts":1,
                                 "mode":"adaptive" # dynamic retry strategy (retry behavior)
                             })
        # Config allows us to customize how a Boto3 AWS client behaves we can the timeouts, retries, etc

        self.s3_client = boto3.client('s3',config=boto_config)
        # creates a low level aws python client for boto3

        self._validate_bucket()
        logger.info(f"S3Storage initialized with bucket: {self.bucket_name} region: {self.region}")

    def _validate_bucket(self) -> None:
        """Check if S3 exists and is accessible
        
        Raises:
            ValueError if bucket doesn't exist
            PermissionError if access denied
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            # checks if s3 bucket exists and verifies credentials to have the permission to access it 

            logger.info(f"S3 bucket {self.bucket_name} is accessible")

        except ClientError as e:
            error_code = e.response['Error']["Code"]

            if error_code == "404":
                raise ValueError(f"s3 bucket: {self.bucket_name} does not exist")

            elif error_code == "403":
                raise PermissionError(f"Access denied to S3 bucket: {self.bucket_name}")

            raise 

    def _get_s3_key(self, document_id: str, file_extension: str, filename: str) -> str:
        """
        Generate S3 key with folder structure.

        Pattern: {doc_type}/{doc_id}/{filename}

        Examples:
            pdf/abc123def456/document.pdf
            pdf/abc123def456/chunks.json
            txt/xyz789/document.txt

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (pdf, txt, md, etc.)
            filename: File name (document.pdf, chunks.json, etc.)

        Returns:
            S3 key string
        """
        return f"{file_extension}/{document_id}/{filename}"

    def _object_exists(self, key: str) -> bool:
        """Check if S3 object exists (using HEAD request)
        Args:
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True 
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False 
            raise 

    def exists(self, document_id: str, file_extension: str) -> bool:
        """
        Check if ALL 4 files (embeddings, metadata, chunks, doc) exist in S3 for this document.

        Returns True only if all files present (all-or-nothing).

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension (pdf, txt, md, etc.)

        Returns:
            True if all 4 files exist
        """
        required_files = [
            f"documents.{file_extension}",
            "chunks.json",
            "embeddings.npy",
            "metadata.json"
        ]

        for filename in required_files:
            key = self._get_s3_key(document_id=document_id,
                                   file_extension=file_extension,
                                   filename=filename)
    
            if not self._object_exists(key):
                logger.debug(f"S3 cache miss for {document_id} (missing: {filename})")
                return False

        logger.debug(f"S3 cache hit for {document_id}")
        return True 

    def save_document(self, document_id: str, file_path: str, file_extension: str) -> None:
        """
        Upload original document to S3.

        Example: s3://bucket/pdf/{doc_id}/document.pdf

        Args:
            document_id: SHA-256 hash of document
            file_path: Path to the uploaded file
            file_extension: File extension (pdf, txt, md, etc.)

        Raises:
            Exception if upload fails
        """
        pass