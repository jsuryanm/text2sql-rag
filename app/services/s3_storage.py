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
        # Config allows us to customize how a Boto3 AWS client behaves on timeouts, retries, etc

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
        key = self._get_s3_key(document_id=document_id, file_extension=file_extension, filename=f"document.{file_extension}")

        try:
            with open(file_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key, # Unique identifier for the object within the bucket
                    Body=f.read(), # Content we are uploading
                    ServerSideEncryption="AES256" # security setting for the data at rest
                )

                # adds or overwrites an object in S3 bucket
            
            logger.info(f"Uploaded original document to S3: {key}")

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise 

    def save_chunks(
        self,
        document_id: str,
        file_extension: str,
        chunks: List[Dict] 
    ) -> None:
        """
        Save chunks to S3 as JSON.

        Example: s3://bucket/pdf/{doc_id}/chunks.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            chunks: List of document chunks

        Raises:
            Exception if upload fails
        """

        key = self._get_s3_key(document_id=document_id,
                               file_extension=file_extension,
                               filename="chunks.json")

        try:
            body = json.dumps(chunks,indent=2).encode('utf-8')

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType="application/json", # defines the format of the file
                ServerSideEncryption='AES256'
            )

            logger.debug(f"Saved {len(chunks)} to S3: {key}")

        except Exception as e:
            logger.error(f"Failed to save chunks to S3: {e}")

    def save_embeddings(self, document_id: str, file_extension: str, embeddings: np.ndarray) -> None:
        """
        Save embeddings to S3 as NumPy binary.

        Example: s3://bucket/pdf/{doc_id}/embeddings.npy

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            embeddings: NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id=document_id,file_extension=file_extension,filename="embeddings.npy")

        try:
            # Serialize np array to bytes (in-memory)
            buffer = io.BytesIO() 
            # this is similar to with open context handler in python but it is faster
            np.save(buffer, embeddings)
            buffer.seek(0)
            # moves the reading pointer back to the original point 

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream',
                # file is being treated as an unknown binary file.
                ServerSideEncryption='AES256'
            )

            logger.debug(f"Saved embeddings {embeddings.shape} to S3: {key}")

        except Exception as e:
            logger.error(f"Failed to save embeddings to S3: {e}")
            raise 


    def save_metadata(self, document_id, file_extension, metadata):
        """
        Save metadata to S3 as JSON.

        Example: s3://bucket/pdf/{doc_id}/metadata.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension
            metadata: Document metadata

        Raises:
            Exception if upload fails
        """
        key = self._get_s3_key(document_id=document_id,
                               file_extension=file_extension,
                               filename="metadata.json")

        try:
            body = json.dumps(metadata, indent=2).encode('utf-8')

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="AES256"
            )

            logger.debug(f"Saved metadata to S3: {key}")

        except Exception as e:
            logger.error(f"Failed to save metadata to S3: {e}")
            raise

    def load_chunks(self, document_id: str, file_extension: str) -> List[Dict]:
        """
        Load chunks from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension to locate correct folder

        Returns:
            List of document chunks

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "chunks.json")

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            # used to retrieve an object and its metadata directly 
            # from AWS S3 bucket into memory

            file_content = response["Body"].read().decode('utf-8')
            # Extract and decode the text content

            chunks = json.loads(file_content)

            logger.debug(f'Loaded {len(chunks)} from S3: {key}')
            return chunks 

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Chunks file not found in S3: {key}")
            raise

    def load_embeddings(self, document_id, file_extension):
        """
        Load embeddings from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension

        Returns:
            NumPy array of shape (num_chunks, 1536)

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "embeddings.npy")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name,
                                                 Key=key)
            buffer = io.BytesIO(response['Body'].read())
            embeddings = np.load(buffer)

            logger.debug(f"Loaded embeddings {embeddings.shape} from S3: {key}")

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                    raise FileNotFoundError(f"Embeddings file not found in S3: {key}")
            raise

    def load_metadata(self, document_id: str, file_extension: str) -> Dict:
        """
        Load metadata from S3.

        Args:
            document_id: SHA-256 hash
            file_extension: File extension

        Returns:
            Document metadata dictionary

        Raises:
            Exception if file not found or load fails
        """
        key = self._get_s3_key(document_id, file_extension, "metadata.json")

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            logger.debug(f"Loaded metadata from S3: {key}")
            return metadata

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Metadata file not found in S3: {key}")
            raise

    def delete(self, document_id: str, file_extension: str) -> None:
        """
        Delete all 4 files for a document from S3.

        Deletes: document.{ext}, chunks.json, embeddings.npy, metadata.json

        Args:
            document_id: SHA-256 hash of document
            file_extension: File extension

        Raises:
            Exception if delete fails
        """
        keys_to_delete = [
            {'Key': self._get_s3_key(document_id, file_extension, f"document.{file_extension}")},
            {'Key': self._get_s3_key(document_id, file_extension, "chunks.json")},
            {'Key': self._get_s3_key(document_id, file_extension, "embeddings.npy")},
            {'Key': self._get_s3_key(document_id, file_extension, "metadata.json")}
        ]

        try:
            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects":keys_to_delete}
            )
            logger.info(f"Deleted S3 cache for document: {document_id}")

        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")
            raise 

    def list_documents(self) -> List[str]:
        """
        List all cached document IDs from S3 across all document types.

        Returns:
            List of document IDs (SHA-256 hashes)

        Note: This scans the entire bucket (uses pagination for >1000 objects)
        """
        document_ids = set()

        try:
            # when we have many files aws returns the results in pages ex, S3 bucket contains 10000 files
            # AWS may not return all 10k at once
            # paginator automatically helps to go through all those pages. it repeatedly calls list_objects_v2 
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" not in page:
                    continue

                for obj in page.get("Contents", []):
                    key_parts = obj['Key'].split('/')

                    if len(key_parts) >= 2:
                        # key_parts[0] = document type (pdf, txt, etc.)
                        # key_parts[1] = document_id
                        document_ids.add(key_parts[1])

                logger.debug(f"Found {len(document_ids)} cached documents in S3")
            return list(document_ids)

        except Exception as e:
            logger.error(f"Failed to list documents from S3: {e}")
            return []

    def get_stats(self) -> Dict:
        """
        Get S3 cache statistics.

        Returns:
            Dictionary with stats (backend, bucket, region, total_documents, total_objects, total_size_mb, documents_by_type)

        Note: This scans the entire bucket to compute stats
        """
        total_size = 0
        total_objects = 0
        doc_type_counts = {}  # Count documents by type (pdf: 10, txt: 5, etc.)

        try:
            # Scan all objects in bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    total_size += obj['Size']
                    total_objects += 1

                    # Count by document type (pdf/, txt/, etc.)
                    doc_type = obj['Key'].split('/')[0] if '/' in obj['Key'] else 'unknown'
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

            stats = {
                "backend": "s3",
                "bucket": self.bucket_name,
                "region": self.region,
                "total_documents": len(self.list_documents()),
                "total_objects": total_objects,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "documents_by_type": doc_type_counts  # Shows pdf: 10, txt: 5, etc.
            }

            logger.info(f"S3 storage stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get S3 stats: {e}")
            return {
                "backend": "s3",
                "bucket": self.bucket_name,
                "error": str(e)
            }




    