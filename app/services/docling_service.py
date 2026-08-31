import logging 
from typing import Any, Dict, List
from pathlib import Path 

from docling_core.transforms.chunker.doc_chunk import DocChunk 

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

def _extract_page_numbers(chunk: DocChunk) -> List[int]:
    """
    Extract page numbers from a Docling chunk.
    Args:
        chunk: A Docling chunk (from HybridChunker.chunk())

    Returns:
        Sorted list of unique page numbers referenced by this chunk
    """
    page_numbers = set()

    if chunk.meta and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:

            for prov in getattr(item, 'prov', None) or None:
                page_no = getattr(prov, 'prov', None)

                if page_no is not None:
                    page_numbers.add(page_no)
    return sorted(page_numbers)

def chunk_with_hybrid(doc, max_tokens: int = 512, min_tokens: int = 256) -> List[Dict[str, Any]]:
    """
    Chunk document using HybridChunker with context awareness.

    Uses OpenAI tokenizer (tiktoken) for consistency with existing embeddings.
    Preserves hierarchical heading context and semantic boundaries.
    Post-processes to merge small chunks for better RAG context.

    Args:
        doc: DoclingDocument from convert_document()
        max_tokens: Maximum tokens per chunk (default: 512)
        min_tokens: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata:
            - text: The chunk text
            - chunk_index: Sequential index
            - token_count: Actual token count
            - start_char: Starting character position
            - end_char: Ending character position
            - headings: List of hierarchical headings (e.g., ["Chapter 1", "Section 1.2"])
            - page_numbers: List of page numbers this chunk spans
            - doc_items: References to original document items
            - captions: Table/figure captions if applicable
    """
    if not DOCLING_AVAILABLE: 
        raise ImportError("Docling is not installed. Run uv pip install docling docling-core")

    try:
        import tiktoken 

        tiktoken_encoder = tiktoken.get_encoding('o200k_base')
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken_encoder,
            max_completion_tokens=max_tokens
        )

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=True # merges chunks with same heading content
        )
        logger.info(f"Chunking with HybridChunker (max_tokens={max_tokens}, merge_peers=True)")

        raw_chunks = list(chunker.chunk(dl_doc=doc))

        logger.info(f"Generated {len(raw_chunks)} raw semantic chunks, merging to target {min_tokens} - {max_tokens} tokens")

        # Post-process: Merge consecutive small chunks to reach target size 
        merged_chunks = []
        current_merged = None 

        for chunk in raw_chunks:
            token_count = len(tiktoken_encoder.encode(chunk.text))

            if current_merged is None:
                # start a new merged chunk
                current_merged = chunk

            else:
                # Check if we can merge with current chunk
                current_tokens = len(tiktoken_encoder.encode(current_merged.text))
                combined_tokens = current_tokens  + token_count

                # Merge if current chunk is undersized and combined won't exceed max
                if current_tokens < min_tokens and combined_tokens <= max_tokens:
                    # merge chunks
                    current_merged.text = current_merged.text + "\n\n" + chunk.text

                    # merge metadata
                    if chunk.meta and chunk.meta.headings:
                        if not current_merged.meta.headings:
                            current_merged.meta.headings = []

                        for h in chunk.meta.headings:
                            if h not in current_merged.meta.headings:
                                current_merged.meta.headings.append(h)

                    # merge page numbers and chunk.meta.origin stores metadata wrt file path, etc
                    if chunk.meta and chunk.meta.doc_items:
                        if not current_merged.meta.doc_items:
                            current_merged.meta.doc_items = []
                        current_merged.meta.doc_items.extend(chunk.meta.doc_items)
                    
                else:
                    # Current chunk is complete, save it and start new one
                    merged_chunks.append(current_merged)
                    current_merged = chunk

        # don't forget last chunk 
        if current_merged is not None:
            merged_chunks.append(current_merged)

        logger.info(f"After merging: {len(merged_chunks)} chunks chunks (avg {sum(len(tiktoken_encoder.encode(c.text)) for c in merged_chunks) / len(merged_chunks):.1f} tokens)")
                
        # Convert to format compatible with existing cache/vector storage
        result = []
        char_position = 0 

        for idx, chunk in enumerate(merged_chunks):
            # Extract heading hierarchy

            headings = []

            if chunk.meta and chunk.meta.headings:
                headings = [h.text for h in chunk.meta.headings if hasattr(h, 'text')]

                page_numbers = _extract_page_numbers(chunk)

                captions = []

                if chunk.meta and hasattr(chunk.meta, 'captions') and chunk.meta.captions:
                    captions = [str(c) for c in chunk.meta.captions]

                doc_items = []
                if chunk.meta and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    doc_items = [str(item)[:100] for item in chunk.meta.doc_items[:3]]

                token_count = len(tiktoken_encoder.encode(chunk.text))

                chunk_text = chunk.text
                start_char = char_position 
                end_char = start_char + len(chunk_text)
                char_position = end_char

                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": idx,
                    "token_index": token_count,
                    "start_char": start_char,
                    "end_char": end_char,
                    "headings": headings,
                    "doc_items": doc_items,
                    "page_numbers": page_numbers,
                    "captions": captions
                }

                result.append(chunk_data)

        if result:
            first_chunk = result[0]
            logger.info(f"Sample chunk metadata - Headings: {first_chunk['headings']}, Pages: {first_chunk['page_numbers']}")

        return result

    except Exception as e:
        logger.error(f"HybridChunker failed: {str(e)}")
        raise Exception(f"Failed to chunk document with HybridChunker")

def parse_and_chunk_document(file_path: str, chunk_size: int = 512, min_chunk_size: int = 256) -> List[int]:
    """
    Parse and chunk document using Docling's context-aware approach.

    This is the main entry point that replaces the old parse_document() + chunk_text() flow.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk (default: 512)
        min_chunk_size: Minimum tokens per chunk - smaller chunks will be merged (default: 256)

    Returns:
        List of chunk dictionaries with rich metadata

    Raises:
        Exception: If both Docling and fallback fail
    """
    if not DOCLING_AVAILABLE:
        logger.info("Docling is not available, cannot use context-aware chunking")
        raise ImportError("Docling is required for context-aware chunking")

    try:
        doc = convert_document(file_path)
        chunks = chunk_with_hybrid(doc, max_tokens=chunk_size, min_tokens=min_chunk_size)

        logger.info(f"Successfully processed {Path(file_path).name}")
        return chunks 

    except Exception as e:
        logger.error(f"Docling processing failed for {Path(file_path)}: {str(e)}")
        raise Exception(f"Failed to process document with Docling: {str(e)}")


def fallback_to_unstructured(file_path: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    Fallback to Unstructured.io for documents Docling cannot handle.

    This maintains compatibility but without context-aware chunking benefits.

    Args:
        file_path: Path to the document file
        chunk_size: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries (without rich metadata)
    """
    logger.warning(f"Using Unstructured.io fallback for {Path(file_path).name}")

    try:
        from app.services.document_service import parse_document, chunk_text

        # Use old token-based chunking
        text = parse_document(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=50)

        # Add empty metadata fields for compatibility
        for chunk in chunks:
            chunk['headings'] = []
            chunk['page_numbers'] = []
            chunk['doc_items'] = []
            chunk['captions'] = []

        logger.info(f"Fallback chunking complete: {len(chunks)} chunks (no context)")

        return chunks

    except Exception as e:
        logger.error(f"Fallback also failed: {str(e)}")
        raise Exception(f"Both Docling and Unstructured failed: {str(e)}")


def get_docling_status() -> Dict[str, Any]:
    """
    Check if Docling is available and functioning.

    Returns:
        Dictionary with status information
    """
    return {
        "docling_available": DOCLING_AVAILABLE,
        "features": {
            "context_aware_chunking": DOCLING_AVAILABLE,
            "heading_preservation": DOCLING_AVAILABLE,
            "table_structure": DOCLING_AVAILABLE,
            "layout_analysis": DOCLING_AVAILABLE
        }
    }

# if __name__ == "__main__":
#     file_path = "/home/surya/multidata-rag/data/transformers_paper.pdf"
#     chunks = parse_and_chunk_document(file_path)
#     print(chunks)