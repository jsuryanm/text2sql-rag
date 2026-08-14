import logging 
from typing import Any, Dict, List
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
            max_tokens=max_tokens
        )

        chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            merge_peers=True 
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
                current_tokens = len(tiktoken_encoder.encode(current_merged))
                combined_tokens = current_tokens  + token_count

                # Merge if current chunk is undersized and combined won't exceed max
                if current_tokens < min_tokens and combined_tokens <= max_tokens:
                    # merge chunks
                    current_tokens.text = current_merged.text + "\n\n" + chunk.text

                    # merge metadata
                    if chunk.meta and chunk.meta.headings:
                        if not current_merged.meta.headings:
                            current_merged.meta.headings = []

                        for h in chunk.meta.headings:
                            if h not in current_merged.meta.headings:
                                current_merged.meta.headings.append(h)

                    # merge page numbers and chunk.meta.origin stores metadata wrt file path, etc
                    if chunk.meta and chunk.meta.doc_items and hasattr(chunk.meta.doc_items, 'prov'):
                        if chunk.meta.origin.page_numbers:
                            if not hasattr(current_merged.meta.origin, 'page_numbers') or not current_merged.meta.origin.page_numbers:
                                if current_merged.meta and current_merged.meta.origin:
                                    current_merged.meta.origin.page_numbers = []
                            if current_merged.meta and current_merged.meta.origin and current_merged.meta.origin.page_numbers is not None:
                                for pn in chunk.meta.origin.page_numbers:
                                    if pn not in current_merged.meta.origin.page_numbers:
                                        current_merged.meta.origin.page_numbers.append(pn)
                    
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




    except Exception as e:
        logger.error(f"HybridChunker failed: {str(e)}")
        raise Exception(f"Failed to chunk document with HybridChunker")
