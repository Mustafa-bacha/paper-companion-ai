# Phase 2: RAG Pipeline Implementation
# Contains: PDF ingestion, chunking, vector indexing, and retrieval-augmented generation

__all__ = [
    'PDFIngestor',
    'ChunkingStrategy',
    'RAGEngine'
]

def __getattr__(name):
    """Lazy import for better performance."""
    if name in ['PDFIngestor', 'ChunkingStrategy']:
        from .ingestion import PDFIngestor, ChunkingStrategy
        return locals()[name]
    elif name == 'RAGEngine':
        from .rag_engine import RAGEngine
        return RAGEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
