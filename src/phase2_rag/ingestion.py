"""
PDF Ingestion Module - Document Processing for RAG Pipeline

This module handles:
1. PDF text extraction with PyMuPDF
2. Section detection (Abstract, Introduction, Methods, Results, etc.)
3. Multiple chunking strategies (fixed-size vs section-based)
4. Vector indexing with FAISS
5. Hybrid retrieval support (BM25 + Dense embeddings)

Features:
- Automatic section detection using regex patterns
- Metadata preservation (paper ID, section, page number)
- Comparison framework for chunking strategies
- GPU-accelerated embeddings with sentence-transformers
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# PDF Processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not installed. PDF processing disabled.")

# LangChain for document handling
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Vector Store
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Vector search disabled.")

# Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: HuggingFace embeddings not installed.")

# BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

import numpy as np
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA CLASSES
# =============================================================================

class ChunkingStrategy(str, Enum):
    """Chunking strategy options."""
    FIXED_SIZE = "fixed_size"
    SECTION_BASED = "section_based"
    RECURSIVE = "recursive"
    HYBRID = "hybrid"


@dataclass
class Section:
    """Represents a section in a paper."""
    name: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    """Represents a parsed PDF paper."""
    paper_id: str
    title: str
    full_text: str
    sections: List[Section]
    page_count: int
    metadata: Dict


@dataclass
class Chunk:
    """Represents a text chunk for indexing."""
    text: str
    paper_id: str
    section: str
    page: int
    chunk_id: str
    metadata: Dict


# =============================================================================
# PDF PARSER
# =============================================================================

class PDFParser:
    """
    Parse PDF files and extract structured content.
    
    Features:
    - Full text extraction
    - Section detection using regex patterns
    - Metadata extraction
    """
    
    # Common section headers in academic papers
    SECTION_PATTERNS = [
        (r'^\s*(?:1\.?\s*)?(?:INTRODUCTION|Introduction)\s*$', 'Introduction'),
        (r'^\s*(?:2\.?\s*)?(?:RELATED\s*WORK|Related\s*Work|BACKGROUND|Background)\s*$', 'Related Work'),
        (r'^\s*(?:3\.?\s*)?(?:METHODOLOGY|Methodology|METHOD|Method|METHODS|Methods|APPROACH|Approach)\s*$', 'Methods'),
        (r'^\s*(?:4\.?\s*)?(?:EXPERIMENTS?|Experiments?|RESULTS?|Results?)\s*$', 'Results'),
        (r'^\s*(?:5\.?\s*)?(?:DISCUSSION|Discussion)\s*$', 'Discussion'),
        (r'^\s*(?:6\.?\s*)?(?:CONCLUSION|Conclusion|CONCLUSIONS|Conclusions)\s*$', 'Conclusion'),
        (r'^\s*(?:ABSTRACT|Abstract)\s*$', 'Abstract'),
        (r'^\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*$', 'References'),
    ]
    
    def __init__(self):
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")
    
    def extract_text(self, pdf_path: Path) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract full text from PDF with page information.
        
        Returns:
            Tuple of (full_text, [(page_num, page_text), ...])
        """
        doc = fitz.open(pdf_path)
        pages = []
        full_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append((page_num + 1, text))
            full_text.append(text)
        
        doc.close()
        return "\n".join(full_text), pages
    
    def detect_sections(self, text: str, pages: List[Tuple[int, str]]) -> List[Section]:
        """
        Detect sections in the paper using regex patterns.
        
        Returns:
            List of Section objects
        """
        sections = []
        lines = text.split('\n')
        
        current_section = "Preamble"
        current_content = []
        current_page_start = 1
        
        line_to_page = {}
        cumulative_lines = 0
        for page_num, page_text in pages:
            page_lines = page_text.split('\n')
            for i, _ in enumerate(page_lines):
                line_to_page[cumulative_lines + i] = page_num
            cumulative_lines += len(page_lines)
        
        for line_idx, line in enumerate(lines):
            matched = False
            
            for pattern, section_name in self.SECTION_PATTERNS:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    # Save previous section
                    if current_content:
                        page_end = line_to_page.get(line_idx - 1, current_page_start)
                        sections.append(Section(
                            name=current_section,
                            content='\n'.join(current_content),
                            page_start=current_page_start,
                            page_end=page_end
                        ))
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    current_page_start = line_to_page.get(line_idx, 1)
                    matched = True
                    break
            
            if not matched:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections.append(Section(
                name=current_section,
                content='\n'.join(current_content),
                page_start=current_page_start,
                page_end=len(pages)
            ))
        
        return sections
    
    def parse(self, pdf_path: Path) -> ParsedPaper:
        """
        Parse a PDF file into structured content.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            ParsedPaper object
        """
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem
        
        # Extract text
        full_text, pages = self.extract_text(pdf_path)
        
        # Detect sections
        sections = self.detect_sections(full_text, pages)
        
        # Extract title (usually first non-empty line)
        title = ""
        for section in sections:
            if section.name == "Preamble":
                lines = section.content.strip().split('\n')
                for line in lines:
                    if line.strip() and len(line.strip()) > 10:
                        title = line.strip()[:200]
                        break
        
        return ParsedPaper(
            paper_id=paper_id,
            title=title,
            full_text=full_text,
            sections=sections,
            page_count=len(pages),
            metadata={
                'source_file': str(pdf_path),
                'page_count': len(pages),
                'section_count': len(sections)
            }
        )


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

class ChunkingEngine:
    """
    Multiple chunking strategies for document processing.
    
    Strategies:
    1. Fixed-size: Simple character-based chunking
    2. Section-based: Chunk by detected sections
    3. Recursive: LangChain's RecursiveCharacterTextSplitter
    4. Hybrid: Section-aware + size limits
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def chunk_fixed_size(self, paper: ParsedPaper) -> List[Chunk]:
        """Simple fixed-size chunking."""
        chunks = []
        text = paper.full_text
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text.strip():
                chunk_id = f"{paper.paper_id}_chunk_{len(chunks)}"
                chunks.append(Chunk(
                    text=chunk_text,
                    paper_id=paper.paper_id,
                    section="Unknown",
                    page=1,
                    chunk_id=chunk_id,
                    metadata={'strategy': 'fixed_size'}
                ))
        
        return chunks
    
    def chunk_section_based(self, paper: ParsedPaper) -> List[Chunk]:
        """Section-aware chunking."""
        chunks = []
        
        for section in paper.sections:
            # Skip references section
            if section.name.lower() == 'references':
                continue
            
            # If section is small enough, keep as one chunk
            if len(section.content) <= self.chunk_size:
                if section.content.strip():
                    chunk_id = f"{paper.paper_id}_{section.name.replace(' ', '_')}_0"
                    chunks.append(Chunk(
                        text=section.content,
                        paper_id=paper.paper_id,
                        section=section.name,
                        page=section.page_start,
                        chunk_id=chunk_id,
                        metadata={'strategy': 'section_based'}
                    ))
            else:
                # Split large sections
                text = section.content
                chunk_num = 0
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk_text = text[i:i + self.chunk_size]
                    if chunk_text.strip():
                        chunk_id = f"{paper.paper_id}_{section.name.replace(' ', '_')}_{chunk_num}"
                        chunks.append(Chunk(
                            text=chunk_text,
                            paper_id=paper.paper_id,
                            section=section.name,
                            page=section.page_start,
                            chunk_id=chunk_id,
                            metadata={'strategy': 'section_based'}
                        ))
                        chunk_num += 1
        
        return chunks
    
    def chunk_recursive(self, paper: ParsedPaper) -> List[Chunk]:
        """LangChain recursive chunking."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for section in paper.sections:
            if section.name.lower() == 'references':
                continue
            
            texts = splitter.split_text(section.content)
            for i, text in enumerate(texts):
                if text.strip():
                    chunk_id = f"{paper.paper_id}_{section.name.replace(' ', '_')}_{i}"
                    chunks.append(Chunk(
                        text=text,
                        paper_id=paper.paper_id,
                        section=section.name,
                        page=section.page_start,
                        chunk_id=chunk_id,
                        metadata={'strategy': 'recursive'}
                    ))
        
        return chunks
    
    def chunk(self, paper: ParsedPaper) -> List[Chunk]:
        """
        Apply chunking strategy to a parsed paper.
        
        Args:
            paper: ParsedPaper object
        
        Returns:
            List of Chunk objects
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self.chunk_fixed_size(paper)
        elif self.strategy == ChunkingStrategy.SECTION_BASED:
            return self.chunk_section_based(paper)
        elif self.strategy == ChunkingStrategy.RECURSIVE:
            return self.chunk_recursive(paper)
        elif self.strategy == ChunkingStrategy.HYBRID:
            # Hybrid: Use section-based but with recursive splitting for large sections
            return self.chunk_recursive(paper)
        else:
            return self.chunk_recursive(paper)


# =============================================================================
# PDF INGESTOR (Main Class)
# =============================================================================

class PDFIngestor:
    """
    Complete PDF ingestion pipeline for RAG.
    
    Features:
    - Batch PDF processing
    - Multiple chunking strategies
    - Vector store creation (FAISS)
    - Hybrid retrieval support (BM25 + Dense)
    """
    
    def __init__(
        self,
        pdf_dir: Path = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_DIR
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        
        # Initialize components
        self.parser = PDFParser() if PYMUPDF_AVAILABLE else None
        self.chunker = ChunkingEngine(chunk_size, chunk_overlap, chunking_strategy)
        
        # Embeddings
        if EMBEDDINGS_AVAILABLE:
            device = 'cuda' if use_gpu else 'cpu'
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': device}
            )
        else:
            self.embeddings = None
        
        # Storage
        self.vector_store = None
        self.bm25_index = None
        self.chunks: List[Chunk] = []
    
    def process_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Process a single PDF file."""
        if not self.parser:
            raise RuntimeError("PDF parser not available. Install PyMuPDF.")
        
        paper = self.parser.parse(pdf_path)
        chunks = self.chunker.chunk(paper)
        
        return chunks
    
    def process_all_pdfs(self, limit: int = None) -> List[Chunk]:
        """
        Process all PDFs in the directory.
        
        Args:
            limit: Maximum number of PDFs to process
        
        Returns:
            List of all chunks
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if limit:
            pdf_files = pdf_files[:limit]
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return []
        
        print(f"📄 Processing {len(pdf_files)} PDFs...")
        
        all_chunks = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error processing {pdf_path}: {e}")
        
        self.chunks = all_chunks
        print(f"✅ Created {len(all_chunks)} chunks from {len(pdf_files)} papers")
        
        return all_chunks
    
    def create_vector_store(self, save_path: Path = None) -> "FAISS":
        """
        Create FAISS vector store from chunks.
        
        Args:
            save_path: Path to save the vector store
        
        Returns:
            FAISS vector store
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-gpu")
        
        if not self.embeddings:
            raise RuntimeError("Embeddings not available. Install langchain-huggingface.")
        
        if not self.chunks:
            print("No chunks to index. Run process_all_pdfs first.")
            return None
        
        print("🔍 Creating vector embeddings...")
        
        # Convert chunks to LangChain documents
        documents = []
        for chunk in tqdm(self.chunks, desc="Preparing documents"):
            doc = Document(
                page_content=chunk.text,
                metadata={
                    'paper_id': chunk.paper_id,
                    'section': chunk.section,
                    'page': chunk.page,
                    'chunk_id': chunk.chunk_id,
                    'strategy': chunk.metadata.get('strategy', 'unknown')
                }
            )
            documents.append(doc)
        
        print("📊 Building FAISS index...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save if path provided
        save_path = save_path or VECTOR_STORE_DIR
        self.vector_store.save_local(str(save_path))
        print(f"✅ Vector store saved to {save_path}")
        
        return self.vector_store
    
    def create_bm25_index(self) -> Optional["BM25Okapi"]:
        """Create BM25 index for hybrid retrieval."""
        if not BM25_AVAILABLE:
            print("BM25 not available. Install with: pip install rank-bm25")
            return None
        
        if not self.chunks:
            print("No chunks to index.")
            return None
        
        print("📚 Creating BM25 index...")
        
        # Tokenize chunks for BM25
        tokenized_corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        print(f"✅ BM25 index created with {len(self.chunks)} documents")
        return self.bm25_index
    
    def load_vector_store(self, path: Path = None) -> "FAISS":
        """Load existing vector store."""
        path = path or VECTOR_STORE_DIR
        
        if not self.embeddings:
            raise RuntimeError("Embeddings not available.")
        
        self.vector_store = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Vector store loaded from {path}")
        return self.vector_store
    
    def search(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results
            use_hybrid: Whether to use hybrid BM25 + dense retrieval
        
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Run create_vector_store first.")
        
        if use_hybrid and self.bm25_index:
            return self._hybrid_search(query, k)
        
        # Pure dense retrieval
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def _hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Search query
            k: Number of results
            alpha: Weight for dense scores (1-alpha for BM25)
        """
        # Get dense results
        dense_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # Get BM25 results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores = bm25_scores / max_bm25
        
        # Combine scores
        combined = {}
        for doc, dense_score in dense_results:
            chunk_id = doc.metadata.get('chunk_id')
            # Lower dense score is better (distance), so we invert
            combined[chunk_id] = {
                'doc': doc,
                'dense': 1 - min(dense_score, 1),  # Normalize and invert
                'bm25': 0
            }
        
        # Add BM25 scores
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id in combined:
                combined[chunk.chunk_id]['bm25'] = bm25_scores[i]
        
        # Compute final scores
        final_results = []
        for chunk_id, scores in combined.items():
            final_score = alpha * scores['dense'] + (1 - alpha) * scores['bm25']
            final_results.append((scores['doc'], 1 - final_score))  # Convert back to distance
        
        # Sort by score
        final_results.sort(key=lambda x: x[1])
        
        return final_results[:k]
    
    def get_stats(self) -> Dict:
        """Get statistics about the ingested data."""
        if not self.chunks:
            return {'status': 'No data ingested'}
        
        papers = set(c.paper_id for c in self.chunks)
        sections = {}
        for c in self.chunks:
            sections[c.section] = sections.get(c.section, 0) + 1
        
        avg_chunk_len = sum(len(c.text) for c in self.chunks) / len(self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_papers': len(papers),
            'sections': sections,
            'avg_chunk_length': avg_chunk_len,
            'chunking_strategy': self.chunking_strategy.value,
            'chunk_size': self.chunk_size,
            'embedding_model': self.embedding_model
        }


# =============================================================================
# CHUNKING STRATEGY COMPARISON
# =============================================================================

def compare_chunking_strategies(pdf_dir: Path = None, limit: int = 5) -> Dict:
    """
    Compare different chunking strategies.
    
    Returns statistics for each strategy.
    """
    pdf_dir = pdf_dir or PDF_DIR
    parser = PDFParser()
    
    # Get sample PDFs
    pdf_files = list(pdf_dir.glob("*.pdf"))[:limit]
    
    if not pdf_files:
        print("No PDF files found.")
        return {}
    
    results = {}
    
    for strategy in ChunkingStrategy:
        chunker = ChunkingEngine(
            chunk_size=500,
            chunk_overlap=50,
            strategy=strategy
        )
        
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                paper = parser.parse(pdf_path)
                chunks = chunker.chunk(paper)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error with {strategy.value} on {pdf_path}: {e}")
        
        if all_chunks:
            results[strategy.value] = {
                'total_chunks': len(all_chunks),
                'avg_chunk_length': sum(len(c.text) for c in all_chunks) / len(all_chunks),
                'min_chunk_length': min(len(c.text) for c in all_chunks),
                'max_chunk_length': max(len(c.text) for c in all_chunks),
                'sections_covered': len(set(c.section for c in all_chunks))
            }
    
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def ingest_pdfs(
    pdf_dir: Path = None,
    index_path: Path = None,
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_hybrid: bool = True
) -> PDFIngestor:
    """
    Main function to ingest PDFs and create vector store.
    
    Args:
        pdf_dir: Directory containing PDFs
        index_path: Path to save vector store
        strategy: Chunking strategy ('fixed_size', 'section_based', 'recursive', 'hybrid')
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        use_hybrid: Whether to also create BM25 index
    
    Returns:
        Configured PDFIngestor instance
    """
    chunking_strategy = ChunkingStrategy(strategy)
    
    ingestor = PDFIngestor(
        pdf_dir=pdf_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy
    )
    
    # Process PDFs
    ingestor.process_all_pdfs()
    
    # Create indexes
    ingestor.create_vector_store(index_path)
    
    if use_hybrid:
        ingestor.create_bm25_index()
    
    # Print stats
    stats = ingestor.get_stats()
    print("\n📊 Ingestion Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return ingestor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    parser.add_argument('--pdf-dir', type=str, default=None, help="Directory with PDFs")
    parser.add_argument('--strategy', choices=['fixed_size', 'section_based', 'recursive', 'hybrid'],
                        default='recursive', help="Chunking strategy")
    parser.add_argument('--chunk-size', type=int, default=500, help="Chunk size")
    parser.add_argument('--compare', action='store_true', help="Compare all strategies")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_chunking_strategies()
        print("\n📊 Chunking Strategy Comparison:")
        for strategy, stats in results.items():
            print(f"\n{strategy}:")
            for key, value in stats.items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    else:
        ingestor = ingest_pdfs(
            pdf_dir=args.pdf_dir,
            strategy=args.strategy,
            chunk_size=args.chunk_size
        )
