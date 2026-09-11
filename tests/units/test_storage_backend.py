"""
Tests for the abstract StorageBackend interface (app/services/storage_backend.py).

StorageBackend is an ABC (Abstract Base Class) - it defines *what* methods a
storage backend must have, but not *how* they work. Both LocalStorageBackend
and S3StorageBackend implement this contract. These tests just check the
contract itself, not any real storage.
"""

import pytest

from app.services.storage_backend import StorageBackend


def test_cannot_instantiate_storage_backend_directly():
    """
    An ABC with @abstractmethod methods can't be instantiated on its own -
    Python raises a TypeError. This is what forces every subclass to
    implement all the required methods.
    """
    with pytest.raises(TypeError):
        StorageBackend()


def test_incomplete_subclass_cannot_be_instantiated():
    """A subclass that skips even one abstract method still can't be built."""

    class IncompleteBackend(StorageBackend):
        def exists(self, document_id, file_extension):
            return False
        # Every other abstract method (save_document, save_chunks, ...) is
        # missing on purpose here.

    with pytest.raises(TypeError):
        IncompleteBackend()


def test_complete_subclass_can_be_instantiated():
    """Once every abstract method is implemented, the subclass works fine."""

    class CompleteBackend(StorageBackend):
        def exists(self, document_id, file_extension):
            return False

        def save_document(self, document_id, file_path, file_extension):
            pass

        def save_chunks(self, document_id, file_extension, chunks):
            pass

        def save_embeddings(self, document_id, file_extension, embeddings):
            pass

        def save_metadata(self, document_id, file_extension, metadata):
            pass

        def load_chunks(self, document_id, file_extension):
            return []

        def load_embeddings(self, document_id, file_extension):
            return None

        def load_metadata(self, document_id, file_extension):
            return {}

        def delete(self, document_id, file_extension):
            pass

        def list_documents(self):
            return []

        def get_stats(self):
            return {}

    backend = CompleteBackend()
    assert backend.exists("doc1", "pdf") is False
