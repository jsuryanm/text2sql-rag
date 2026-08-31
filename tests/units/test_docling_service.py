import sys 
import types 

from types import SimpleNamespace
"""
This is a lightweight container obj that allows you to read,
write and delete the attributes using the dot notation, 
without writing an empty class

from types import SimpleNamespace

# Create the object instantly
data = SimpleNamespace(id=101, name="Alice")

# Access and modify attributes with dot notation
print(data.name)  # Output: Alice
data.role = "Admin"  # Dynamically add new attributes
"""
from pathlib import Path

from app.services import docling_service as module

import pytest

def test_get_docling_status():
    result = module.get_docling_status()
    assert isinstance(result, dict)
    assert "docling_available" in result
    assert "features" in result 

def test_convert_document_file_not_found():
    """Check if FileNotFoundError is not raised """
    with pytest.raises(FileNotFoundError):
        # pytest.raises(FileNotFoundError) will expect FileNotFoundError to be raised
        module.convert_document("this_file_doesnt_exist.pdf")

def test_convert_document_success(monkeypatch, tmp_path: Path):
    """Create a mock DocumentConverter object and run the convert function, we get a fake result
    .monkeypatch temporarily replaces something in the application while test is running.
    
    For example, your application has:
    DOCLING_AVAILABLE = True

    We could temporarily change it:

    monkeypatch.setattr(module, "DOCLING_AVAILABLE", False)
    
    only for this test. After this test it resets."""

    file_path = tmp_path / "example.pdf"
    file_path.write_text("fake pdf content") 

    # creates a fake conversion result
    fake_doc = SimpleNamespace(
        texts=["Hello", "World"]
    )

    fake_result = SimpleNamespace(
        document=fake_doc
    )

    # Create a fake DocumentConverter 
    class FakeConverter:
        def convert(self, path):
            return fake_result

    # Make Docling look available 
    monkeypatch.setattr(module, "DOCLING_AVAILABLE", True)

    # Replace the real DocumentConverter with a fake 
    monkeypatch.setattr(module, "DocumentConverter", FakeConverter)

    result = module.convert_document(str(file_path))

    assert result is fake_doc

def test_chunk_with_hybrid(monkeypatch):

    # ---------------------------------
    # 1. Fake tokenizer/encoder
    # ---------------------------------

    class FakeEncoder:

        def encode(self, text):
            return text.split()

    fake_encoder = FakeEncoder()


    # ---------------------------------
    # 2. Fake tiktoken module
    # ---------------------------------

    fake_tiktoken = types.ModuleType("tiktoken")

    fake_tiktoken.get_encoding = lambda name: fake_encoder

    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        fake_tiktoken
    )


    # ---------------------------------
    # 3. Fake OpenAI tokenizer
    # ---------------------------------

    class FakeOpenAITokenizer:

        def __init__(self, tokenizer, max_completion_tokens):
            self.tokenizer = tokenizer
            self.max_completion_tokens = max_completion_tokens


    monkeypatch.setattr(
        module,
        "OpenAITokenizer",
        FakeOpenAITokenizer
    )


    # ---------------------------------
    # 4. Create a fake chunk
    # ---------------------------------

    fake_heading = SimpleNamespace(
        text="Introduction"
    )

    fake_item = SimpleNamespace(
        prov=[SimpleNamespace(prov=1)]
    )

    fake_meta = SimpleNamespace(
        headings=[fake_heading],
        doc_items=[fake_item],
        captions=["Example caption"]
    )

    fake_chunk = SimpleNamespace(
        text="Hello world",
        meta=fake_meta
    )


    # ---------------------------------
    # 5. Fake HybridChunker
    # ---------------------------------

    class FakeHybridChunker:

        def __init__(
            self,
            tokenizer,
            max_tokens,
            merge_peers
        ):
            pass

        def chunk(self, dl_doc):
            return [fake_chunk]


    monkeypatch.setattr(
        module,
        "HybridChunker",
        FakeHybridChunker
    )


    # ---------------------------------
    # 6. Run the REAL function
    # ---------------------------------

    monkeypatch.setattr(
        module,
        "DOCLING_AVAILABLE",
        True
    )

    result = module.chunk_with_hybrid(
        doc=SimpleNamespace(),
        max_tokens=10,
        min_tokens=1
    )


    # ---------------------------------
    # 7. Check the result
    # ---------------------------------

    assert len(result) == 1

    assert result[0]["text"] == "Hello world"

    assert result[0]["chunk_index"] == 0

    assert result[0]["token_index"] == 2

    assert result[0]["headings"] == ["Introduction"]

    assert result[0]["page_numbers"] == [1]

    assert result[0]["captions"] == ["Example caption"]    