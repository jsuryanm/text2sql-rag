import logging 
from typing import Any, Dict, Any 
from pathlib import Path 

logger = logging.getLogger("rag_app.docling_service")

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker 
    # HybridChunker inspects docs structure, then uses token counts to make those 
    # chunks fit model's token limit and finally tries to merge small neighboring chunks 
    # when its safe 
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    DOCLING_AVAILABLE = True 

except ImportError as e:
    logger.warning(f"Docling is not available: {e}")
    DOCLING_AVAILABLE = False 


def convert_document(file_path: str):
    """
    Convert document using Docling's advanced layout analysis.

    Args:
        file_path: Path to the document file

    Returns:
        DoclingDocument: Structured document with hierarchy preserved

    Raises:
        ImportError: If Docling is not installed
        Exception: If conversion fails
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not available. Please install docling docling-core")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        logger.info(f"Converting document with Docling: {Path(file_path).name}")

        converter = DocumentConverter()
        # parses document supporting different formats returns a docling document
        result = converter.convert(file_path)
        doc = result.document # contains fields like tables, text, images, etc

        logger.info(f"Document converted into DoclingDocument")
        logger.info(f"Docling Document has :{len(doc.texts)} texts")

        return doc 

    except Exception as e:
        logger.error(f"Docling conversion failed: {str(e)}")
        raise Exception(f"Failed to convert document with Docling: {str(e)}")

def chunk_with_hybrid(doc, max_tokens: int = 512,):
    pass

