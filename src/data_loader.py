"""
Data Loader Module - Automated Data Collection from ArXiv

This module handles:
1. Fetching paper metadata (titles, abstracts, categories) from ArXiv API
2. Downloading PDFs for Phase 2 RAG pipeline
3. Saving structured data for Phase 1 training

Reference: https://arxiv.org/help/api/
"""

import arxiv
import os
import json
import re
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
ABSTRACTS_FILE = DATA_DIR / "abstracts.jsonl"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def map_category_to_topic(categories: List[str]) -> str:
    """
    Map ArXiv categories to simplified topics for classification.
    
    Categories:
    - NLP: cs.CL (Computation and Language)
    - CV: cs.CV (Computer Vision)
    - Security: cs.CR (Cryptography and Security)
    - Healthcare: cs.AI + q-bio.*, medical keywords
    - ML: cs.LG, stat.ML (Machine Learning)
    - Other: everything else
    """
    category_map = {
        'cs.CL': 'NLP',
        'cs.CV': 'CV',
        'cs.CR': 'Security',
        'cs.LG': 'ML',
        'stat.ML': 'ML',
        'cs.AI': 'AI',
        'cs.NE': 'ML',  # Neural and Evolutionary Computing
        'cs.IR': 'NLP',  # Information Retrieval
    }
    
    primary = categories[0] if categories else 'cs.AI'
    
    # Check for healthcare/bio categories
    for cat in categories:
        if cat.startswith('q-bio') or cat.startswith('physics.med'):
            return 'Healthcare'
    
    return category_map.get(primary, 'Other')


def fetch_arxiv_data(
    query: str = "cat:cs.CL AND (LLM OR Transformer OR NLP)",
    max_results: int = 100,
    download_pdfs: bool = False,
    append_mode: bool = False
) -> List[Dict]:
    """
    Fetch papers from ArXiv API.
    
    Args:
        query: ArXiv search query (supports boolean operators)
               Examples:
               - "cat:cs.CL" - All NLP papers
               - "cat:cs.CV AND deep learning" - CV papers about deep learning
               - "cat:cs.AI AND (RAG OR retrieval)" - AI papers about RAG
        max_results: Maximum number of papers to fetch
        download_pdfs: If True, downloads PDFs for RAG pipeline
        append_mode: If True, appends to existing abstracts file
    
    Returns:
        List of paper metadata dictionaries
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    print(f"🔍 Fetching up to {max_results} papers for query: {query}")
    
    papers = []
    mode = 'a' if append_mode else 'w'
    
    with open(ABSTRACTS_FILE, mode, encoding='utf-8') as f:
        for result in tqdm(client.results(search), total=max_results, desc="Fetching"):
            # Extract paper ID from entry_id URL
            paper_id = result.entry_id.split('/')[-1]
            
            # Clean abstract (remove newlines for easier processing)
            abstract = result.summary.replace("\n", " ").strip()
            
            # Map to simplified topic
            topic = map_category_to_topic(result.categories)
            
            # Build metadata
            meta = {
                "id": paper_id,
                "title": result.title.replace("\n", " ").strip(),
                "abstract": abstract,
                "categories": list(result.categories),
                "topic": topic,  # Simplified label for classification
                "published": result.published.isoformat(),
                "authors": [author.name for author in result.authors[:5]],  # First 5 authors
                "pdf_url": result.pdf_url
            }
            
            # Save metadata
            json.dump(meta, f)
            f.write('\n')
            papers.append(meta)
            
            # Download PDF if requested
            if download_pdfs:
                pdf_filename = f"{sanitize_filename(paper_id)}.pdf"
                pdf_path = PDF_DIR / pdf_filename
                
                if not pdf_path.exists():
                    try:
                        result.download_pdf(dirpath=str(PDF_DIR), filename=pdf_filename)
                    except Exception as e:
                        print(f"⚠️ Failed to download {paper_id}: {e}")
    
    print(f"✅ Done! {len(papers)} abstracts saved to {ABSTRACTS_FILE}")
    if download_pdfs:
        pdf_count = len(list(PDF_DIR.glob("*.pdf")))
        print(f"📄 {pdf_count} PDFs available in {PDF_DIR}")
    
    return papers


def fetch_diverse_dataset(
    topics: List[str] = None,
    papers_per_topic: int = 50,
    download_pdfs: bool = False
) -> Dict[str, List[Dict]]:
    """
    Fetch a diverse dataset with balanced topics for classification training.
    
    Args:
        topics: List of topic queries. If None, uses default diverse set.
        papers_per_topic: Number of papers to fetch per topic
        download_pdfs: Whether to download PDFs
    
    Returns:
        Dictionary mapping topics to paper lists
    """
    if topics is None:
        topics = {
            'NLP': 'cat:cs.CL',
            'CV': 'cat:cs.CV',
            'Security': 'cat:cs.CR',
            'ML': 'cat:cs.LG AND (neural OR deep learning)',
            'Healthcare': '(cat:cs.AI OR cat:cs.LG) AND (medical OR healthcare OR clinical)'
        }
    
    all_papers = {}
    
    # Clear existing file for fresh balanced dataset
    open(ABSTRACTS_FILE, 'w').close()
    
    for topic_name, query in topics.items():
        print(f"\n📚 Fetching {topic_name} papers...")
        papers = fetch_arxiv_data(
            query=query,
            max_results=papers_per_topic,
            download_pdfs=download_pdfs,
            append_mode=True
        )
        
        # Override topic with our label
        for paper in papers:
            paper['topic'] = topic_name
        
        all_papers[topic_name] = papers
    
    # Rewrite with correct labels
    with open(ABSTRACTS_FILE, 'w', encoding='utf-8') as f:
        for topic_papers in all_papers.values():
            for paper in topic_papers:
                json.dump(paper, f)
                f.write('\n')
    
    total = sum(len(p) for p in all_papers.values())
    print(f"\n🎉 Total: {total} papers across {len(all_papers)} topics")
    
    return all_papers


def load_abstracts(file_path: Optional[Path] = None) -> List[Dict]:
    """
    Load abstracts from JSONL file.
    
    Args:
        file_path: Path to abstracts file. Uses default if None.
    
    Returns:
        List of paper metadata dictionaries
    """
    file_path = file_path or ABSTRACTS_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"Abstracts file not found: {file_path}")
    
    papers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    
    return papers


def get_dataset_stats(papers: List[Dict] = None) -> Dict:
    """
    Get statistics about the dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    if papers is None:
        papers = load_abstracts()
    
    topics = {}
    total_abstract_length = 0
    
    for paper in papers:
        topic = paper.get('topic', 'Unknown')
        topics[topic] = topics.get(topic, 0) + 1
        total_abstract_length += len(paper.get('abstract', ''))
    
    return {
        'total_papers': len(papers),
        'topics': topics,
        'avg_abstract_length': total_abstract_length / len(papers) if papers else 0,
        'pdf_count': len(list(PDF_DIR.glob("*.pdf")))
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXiv Data Loader")
    parser.add_argument('--mode', choices=['single', 'diverse'], default='diverse',
                        help="'single' for one query, 'diverse' for balanced multi-topic")
    parser.add_argument('--query', type=str, default="cat:cs.CL AND Transformer",
                        help="ArXiv query for single mode")
    parser.add_argument('--max-results', type=int, default=100,
                        help="Max papers to fetch (per topic for diverse mode)")
    parser.add_argument('--download-pdfs', action='store_true',
                        help="Download PDFs for RAG pipeline")
    
    args = parser.parse_args()
    
    if args.mode == 'diverse':
        papers = fetch_diverse_dataset(
            papers_per_topic=args.max_results // 5,  # Split across 5 topics
            download_pdfs=args.download_pdfs
        )
    else:
        papers = fetch_arxiv_data(
            query=args.query,
            max_results=args.max_results,
            download_pdfs=args.download_pdfs
        )
    
    # Show statistics
    stats = get_dataset_stats()
    print("\n📊 Dataset Statistics:")
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   Topics: {stats['topics']}")
    print(f"   Avg abstract length: {stats['avg_abstract_length']:.0f} chars")
    print(f"   PDFs downloaded: {stats['pdf_count']}")
