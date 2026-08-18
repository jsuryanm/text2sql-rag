from typing import List, Dict, Any
import tiktoken
import logging 
from pathlib import Path

from unstructured.partition.auto import partition
# automatically detects file type and breaks the document down into a list of structured text elements (like titles, headers, and paragraphs)
from langchain_text_splitters import TokenTextSplitter

logger = logging.getLogger("rag_app.document_service")

def parse_document(file_path: str) -> str:
    """
    Parse any document type and return extracted text.
    Uses fast direct read for simple text files (.txt, .md, .csv).
    Uses Unstructured.io for complex formats (PDF, DOCX, JSON, etc.).

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted text content from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = Path(file_path).suffix.lower()

    if file_extension in ['.txt','.csv','.log','.json']:
        try:
            logger.info(f"Using fast text reader for {file_extension} file")

            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        except UnicodeDecodeError:
            # try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Fast text read failed: {e}, falling back to unstructured")

        except Exception as e:
            logger.warning(f"Fast text read failed: {e}, falling back to unstructured")

    try:
        # Using unstructured for parsing (pdf, docx, etc)
        logger.info(f"Using unstructured library for: {file_extension} file")
        elements = partition(
            filename=file_path,
            strategy='fast' # fast mode no OCR works without tesseract
        )

        text = "\n\n".join([str(el) for el in elements])

        return text

    except Exception as e:
        raise Exception(f"Failed to parse documents {file_path}: {str(e)}")

def chunk_text(
    text: str,
    chunk_size: int = 512, 
    overlap: int = 50,
    encoding_name: str = "o200k_base" # GPT-4 encoding
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing:
            - text: The chunk text
            - chunk_index: Index of the chunk
            - token_count: Number of tokens in the chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    try:
        tokenizer = tiktoken.get_encoding(encoding_name=encoding_name)
        # this tokenizer is used for counting the no of tokens

    except Exception as e:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini") 

    splitter = TokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True 
        # for every chunk it will record where exactly the chunk starts in the original document 
    )

    docs = splitter.create_documents([text])

    chunks = []

    for doc in docs:
        chunk_str = doc.page_content 
        start_char = doc.metadata.get("start_index", 0)

        chunk_data = {
            "text": chunk_str, 
            "chunk_index": len(chunks),
            "token_count": len(tokenizer.encode(chunk_str)),
            "start_char": start_char,
            "end_char": start_char + len(chunk_str)
        }

        chunks.append(chunk_data)

    return chunks

def get_document_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a document.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with document statistics
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = parse_document(file_path)

    tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')

    tokens = tokenizer.encode(text)

    return {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_type": path.suffix,
        "token_count": len(tokens),
        "estimated_chunks_512": (len(tokens) // 512) + 1
    }

def parse_and_chunk_with_context(file_path: str, chunk_size: int = 512, min_chunk_size: int = 256) -> List[Dict[str, Any]]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the RECOMMENDED method that provides:
    - Semantic boundary detection (no mid-sentence splits)
    - Hierarchical heading context preservation
    - Rich metadata (page numbers, captions, document structure)
    - Smart merging to ensure chunks are 256-512 tokens (not too small)

    Falls back to traditional token-based chunking if Docling is unavailable.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata
    """
    file_extension = Path(file_path).suffix.lower()

    if file_extension in [".txt", ".md", ".csv", ".log", ".json"]:
        logger.info(f"Using fast token-based chunking for {file_extension}")
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk['heading'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.info(f"Fast chunking complete: {len(chunks)} chunks")
        return chunks 

    try:
        from app.services.docling_service import parse_and_chunk_document

        logger.info(f"Using Docling for context-aware chunking: {Path(file_path).name}")
        chunks = parse_and_chunk_document(file_path, chunk_size=chunk_size, min_chunk_size=min_chunk_size)

        logger.info(f"Docling chunking complete: {len(chunks)} with heading context")
        return chunks

    except ImportError as e: 
        logger.warning(f"Docling not available , falling back to token based chunking: {e}")

        text = parse_document(file_path)
        chunks = chunk_text(text=text,chunk_size=chunk_size,overlap=50)

        # add empty metadata fields
        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.info(f'Token based chunking complete: {len(chunks)} (no context)')
        return chunks 

    except Exception as e:
        logger.error(f"Docling failed, falling back to token-based chunking: {e}")

        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size, overlap=50)

        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.warning(f"Using fallback chunking: {len(chunks)} chunks (no context)")
        return chunks

