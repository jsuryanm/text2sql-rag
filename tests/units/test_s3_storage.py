"""
Tests for S3StorageBackend (app/services/s3_storage.py).

This backend talks to real AWS S3 in production. We obviously don't want
tests hitting real AWS (slow, costs money, needs real credentials). Instead
we use `moto`, a library that intercepts boto3 calls and fakes S3 entirely
in memory. From the code's point of view, it looks exactly like real S3.

The `mock_aws` decorator below is what turns on that fake AWS for a test.
"""

import io

import numpy as np
import pytest
import boto3
from moto import mock_aws

from app.services.s3_storage import S3StorageBackend

BUCKET_NAME = "test-rag-bucket"
REGION = "us-east-1"


@pytest.fixture
def aws_credentials(monkeypatch):
    """
    boto3 looks for AWS credentials even when moto is faking the API calls.
    Setting fake env vars here keeps boto3 happy without touching real AWS.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


@pytest.fixture
def s3_storage(aws_credentials):
    """
    Create a fake S3 bucket, then hand back an S3StorageBackend pointed at it.

    `with mock_aws():` starts the fake AWS world; everything inside this
    fixture (bucket creation, and later, whatever the test does with
    s3_storage) runs against the fake S3, not the real one.
    """
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET_NAME)

        yield S3StorageBackend(bucket_name=BUCKET_NAME)


class TestS3StorageBackend:
    def test_initialization_validates_bucket(self, s3_storage):
        assert s3_storage.bucket_name == BUCKET_NAME

    def test_initialization_fails_for_missing_bucket(self, aws_credentials):
        with mock_aws():
            # No bucket created this time - S3StorageBackend should refuse to start.
            with pytest.raises(ValueError):
                S3StorageBackend(bucket_name="bucket-that-does-not-exist")

    def test_save_and_load_document(self, s3_storage, temp_document):
        doc_id = "doc123"
        s3_storage.save_document(doc_id, temp_document, "pdf")

        key = s3_storage._get_s3_key(doc_id, "pdf", "document.pdf")
        assert s3_storage._object_exists(key)

    def test_save_and_load_chunks(self, s3_storage, sample_chunks):
        doc_id = "doc123"
        s3_storage.save_chunks(doc_id, "pdf", sample_chunks)

        loaded = s3_storage.load_chunks(doc_id, "pdf")
        assert loaded == sample_chunks

    def test_save_and_load_embeddings(self, s3_storage, sample_embeddings):
        doc_id = "doc123"
        s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)

        loaded = s3_storage.load_embeddings(doc_id, "pdf")
        assert np.array_equal(loaded, sample_embeddings)

    def test_save_and_load_metadata(self, s3_storage, sample_metadata):
        doc_id = "doc123"
        s3_storage.save_metadata(doc_id, "pdf", sample_metadata)

        loaded = s3_storage.load_metadata(doc_id, "pdf")
        assert loaded == sample_metadata

    def test_load_chunks_missing_raises_file_not_found(self, s3_storage):
        with pytest.raises(FileNotFoundError):
            s3_storage.load_chunks("doc-that-does-not-exist", "pdf")

    def test_exists_false_when_nothing_saved(self, s3_storage):
        assert s3_storage.exists("doc123", "pdf") is False

    def test_exists_true_once_all_four_files_saved(
        self, s3_storage, temp_document, sample_chunks, sample_embeddings, sample_metadata
    ):
        doc_id = "doc123"

        # exists() requires document + chunks + embeddings + metadata all present.
        s3_storage.save_document(doc_id, temp_document, "pdf")
        s3_storage.save_chunks(doc_id, "pdf", sample_chunks)
        s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        s3_storage.save_metadata(doc_id, "pdf", sample_metadata)

        assert s3_storage.exists(doc_id, "pdf") is True

    def test_delete_removes_all_files(
        self, s3_storage, temp_document, sample_chunks, sample_embeddings, sample_metadata
    ):
        doc_id = "doc123"
        s3_storage.save_document(doc_id, temp_document, "pdf")
        s3_storage.save_chunks(doc_id, "pdf", sample_chunks)
        s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        s3_storage.save_metadata(doc_id, "pdf", sample_metadata)
        assert s3_storage.exists(doc_id, "pdf") is True

        s3_storage.delete(doc_id, "pdf")

        assert s3_storage.exists(doc_id, "pdf") is False

    def test_list_documents(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata):
        for i in range(2):
            doc_id = f"doc_{i}"
            s3_storage.save_chunks(doc_id, "pdf", sample_chunks)
            s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
            s3_storage.save_metadata(doc_id, "pdf", sample_metadata)

        doc_list = s3_storage.list_documents()
        assert set(doc_list) == {"doc_0", "doc_1"}

    def test_get_stats(self, s3_storage, sample_chunks, sample_embeddings, sample_metadata):
        s3_storage.save_chunks("doc_1", "pdf", sample_chunks)
        s3_storage.save_embeddings("doc_1", "pdf", sample_embeddings)
        s3_storage.save_metadata("doc_1", "pdf", sample_metadata)

        stats = s3_storage.get_stats()

        assert stats["backend"] == "s3"
        assert stats["bucket"] == BUCKET_NAME
        assert stats["total_objects"] == 3
