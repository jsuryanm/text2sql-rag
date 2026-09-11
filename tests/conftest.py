"""
conftest.py (Shared pytest setup)
Pytest automatically loads this file before it collects/runs test. 
Its a standard place: 
    - fixtures shared across multiple test files.
    - one-time setup that must happen before you test files import the application code they're testing 
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def _register_fake_module(dotted_name: str, **attrs) -> types.ModuleType:
    """Insert a fake module into sys.modules """


# Shared fixtures below are available to every test file, without importing
# anything - pytest auto-discovers fixtures defined in conftest.py.

@pytest.fixture
def sample_chunks():
    """A tiny, realistic list of document chunks."""
    return [
        {"text": "This is the first chunk of text.", "metadata": {"page": 1, "tokens": 7}},
        {"text": "This is the second chunk of text.", "metadata": {"page": 1, "tokens": 7}},
    ]


@pytest.fixture
def sample_embeddings():
    """Fake embedding vectors matching sample_chunks (2 rows x 1536 dims)."""
    return np.random.rand(2, 1536).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Fake document metadata."""
    return {
        "filename": "test_document.pdf",
        "file_extension": "pdf",
        "num_chunks": 2,
    }


@pytest.fixture
def temp_document(tmp_path):
    """Create a temporary test document file on disk."""
    doc_path = tmp_path / "test_document.pdf"
    doc_path.write_text("This is a test document content.")
    return doc_path
