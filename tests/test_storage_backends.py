import pytest
import tempfile
import shutil 
import numpy as np

from pathlib import Path 
from unittest.mock import patch, MagicMock

from app.services.local_storage import LocalStorageBackend

# pytest.fixture is a decorator used to define reusable setup and teardown code for your test cases
@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing"""
    return [
        {
            "text": "This is the first chunk of text.",
            "metadata": {"page": 1, "tokens": 7}
        }
    ]

@pytest.fixture
def sample_embeddings():
    """Sample embeddings array (2 chunks x 1536 dimensions)."""
    return np.random.rand(2, 1536).astype(np.float32)

@pytest.fixture
def temp_document(tmp_path):
    """Create a temporary test document file."""
    doc_path = tmp_path / "test_document.pdf"
    doc_path.write_text("This is a test document content.")
    return doc_path


class TestLocalStorageBackend:
    """Tests for local filesystem storage backend."""

    @pytest.fixture
    def local_storage(self, tmp_path):
        """Create a LocalStorageBackend with temporary directory."""
        return LocalStorageBackend(cache_dir=tmp_path)

    def test_initialization(self, local_storage, tmp_path):
        """Test that LocalStorageBackend initializes correctly"""
        assert local_storage.cache_dir == tmp_path
        assert local_storage.cache_dir.exists()

    def test_save_and_load_chunks(self, local_storage, sample_chunks):
        """Test saving and loading chunks.json."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save chunks
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Verify file exists
        chunks_file = local_storage._get_document_path(doc_id) / "chunks.json"
        assert chunks_file.exists()

        # Load and verify
        loaded_chunks = local_storage.load_chunks(doc_id, file_extension)
        assert loaded_chunks == sample_chunks

    def test_save_and_load_embeddings(self, local_storage, sample_embeddings):
        """Test saving and loading embeddings.npy."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save embeddings
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)

        # Verify file exists
        embeddings_file = local_storage._get_document_path(doc_id) / "embeddings.npy"
        assert embeddings_file.exists()

        # Load and verify
        loaded_embeddings = local_storage.load_embeddings(doc_id, file_extension)
        assert np.array_equal(loaded_embeddings, sample_embeddings)

    def test_save_and_load_metadata(self, local_storage, sample_metadata):
        """Test saving and loading metadata.json."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save metadata
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Verify file exists
        metadata_file = local_storage._get_document_path(doc_id) / "metadata.json"
        assert metadata_file.exists()

        # Load and verify
        loaded_metadata = local_storage.load_metadata(doc_id, file_extension)
        assert loaded_metadata == sample_metadata

    def test_save_and_load_document(self, local_storage, temp_document):
        """Test saving and loading original document."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save document
        local_storage.save_document(doc_id, temp_document, file_extension)

        # Verify file exists
        saved_doc = local_storage._get_document_path(doc_id) / f"document.{file_extension}"
        assert saved_doc.exists()

        # Verify content
        assert saved_doc.read_text() == temp_document.read_text()

    def test_exists_all_files(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test exists() returns True when all files present."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Initially should not exist
        assert not local_storage.exists(doc_id, file_extension)

        # Save all files
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Now should exist
        assert local_storage.exists(doc_id, file_extension)

    def test_exists_partial_files(self, local_storage, sample_chunks):
        """Test exists() returns False when only some files present."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save only chunks
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)

        # Should not exist (missing embeddings and metadata)
        assert not local_storage.exists(doc_id, file_extension)

    def test_delete(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test deleting all files for a document."""
        doc_id = "test_doc_123"
        file_extension = "pdf"

        # Save all files
        local_storage.save_chunks(doc_id, file_extension, sample_chunks)
        local_storage.save_embeddings(doc_id, file_extension, sample_embeddings)
        local_storage.save_metadata(doc_id, file_extension, sample_metadata)

        # Verify exists
        assert local_storage.exists(doc_id, file_extension)

        # Delete
        local_storage.delete(doc_id, file_extension)

        # Verify deleted
        assert not local_storage.exists(doc_id, file_extension)
        assert not local_storage._get_document_path(doc_id).exists()

    def test_list_documents(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test listing all cached documents."""
        # Initially empty
        assert local_storage.list_documents() == []

        # Add multiple documents
        for i in range(3):
            doc_id = f"doc_{i}"
            local_storage.save_chunks(doc_id, "pdf", sample_chunks)
            local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
            local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        # Verify list
        doc_list = local_storage.list_documents()
        assert len(doc_list) == 3
        assert "doc_0" in doc_list
        assert "doc_1" in doc_list
        assert "doc_2" in doc_list

    def test_get_stats(self, local_storage, sample_chunks, sample_embeddings, sample_metadata):
        """Test getting storage statistics."""
        # Add a document
        doc_id = "test_doc"
        local_storage.save_chunks(doc_id, "pdf", sample_chunks)
        local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        # Get stats
        stats = local_storage.get_stats()

        assert stats["backend"] == "local"
        assert stats["cache_dir"] == str(local_storage.cache_dir)
        assert stats["total_documents"] == 1
        assert stats["total_files"] == 3  # chunks, embeddings, metadata
        assert stats["total_size_mb"] > 0

