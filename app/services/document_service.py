from typing import List, Dict, Any
import tiktoken
import logging 
from pathlib import Path

from unstructured.partition.auto import partition
# allows users to extract like titles, narrative text and tables
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
        encoding_name: str = "cl100k_base" # GPT-4 encoding
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
    pass