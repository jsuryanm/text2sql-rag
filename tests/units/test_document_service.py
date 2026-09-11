"""
Tests for document_service.py (parse_document, chunk_text, get_document_stats,
parse_and_chunk_with_context).

These are plain functions (no class), so tests just call them directly. We
stick to .txt/.csv files here because those go through the fast native-read
path (no `unstructured` parsing library involved), which keeps these tests
quick and dependency-free.
"""

import pytest

from app.services.document_service import (
    parse_document,
    chunk_text,
    get_document_stats,
    parse_and_chunk_with_context,
)


class TestParseDocument:
    def test_reads_txt_file_content(self, tmp_path):
        file_path = tmp_path / "notes.txt"
        file_path.write_text("Hello, this is a test document.")

        text = parse_document(str(file_path))

        assert text == "Hello, this is a test document."

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_document(str(tmp_path / "missing.txt"))


class TestChunkText:
    def test_short_text_produces_one_chunk(self):
        chunks = chunk_text("This is a short sentence.", chunk_size=512, overlap=50)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short sentence."
        assert chunks[0]["chunk_index"] == 0

    def test_long_text_produces_multiple_chunks(self):
        # Repeat a sentence enough times to force the splitter over chunk_size tokens.
        long_text = "The quick brown fox jumps over the lazy dog. " * 200

        chunks = chunk_text(long_text, chunk_size=50, overlap=5)

        assert len(chunks) > 1
        # chunk_index should be sequential starting at 0
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_each_chunk_reports_a_token_count(self):
        chunks = chunk_text("hello world", chunk_size=512, overlap=50)
        assert chunks[0]["token_count"] > 0


class TestGetDocumentStats:
    def test_reports_basic_file_info(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Some sample text content for stats.")

        stats = get_document_stats(str(file_path))

        assert stats["filename"] == "sample.txt"
        assert stats["file_type"] == ".txt"
        assert stats["token_count"] > 0
        assert stats["file_size_bytes"] == file_path.stat().st_size

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_document_stats(str(tmp_path / "missing.txt"))


class TestParseAndChunkWithContext:
    def test_txt_file_uses_fast_path_and_adds_empty_metadata(self, tmp_path):
        file_path = tmp_path / "sample.txt"
        file_path.write_text("Some content to chunk.")

        chunks = parse_and_chunk_with_context(str(file_path))

        assert len(chunks) >= 1
        # Fast path chunks should have the same empty-metadata fields the
        # Docling path would add, so downstream code can treat them uniformly.
        assert chunks[0]["page_numbers"] == []
        assert chunks[0]["doc_items"] == []
        assert chunks[0]["captions"] == []
