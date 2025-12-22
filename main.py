#!/usr/bin/env python3
"""
Research Paper Companion AI - Main CLI Entry Point

A comprehensive research assistant that:
- Indexes open-access papers from ArXiv
- Implements transformer variants from scratch (Phase 1)
- Provides RAG-based Q&A with citations (Phase 2)
- Generates related work paragraphs

Usage:
    python main.py --mode <mode> [options]

Modes:
    download    - Download papers from ArXiv
    train       - Train Phase 1 transformer models
    ingest      - Ingest PDFs for RAG pipeline
    ask         - Ask questions (RAG mode)
    interactive - Interactive Q&A session
    evaluate    - Run evaluation metrics
    compare     - Compare chunking strategies

Examples:
    # Download papers
    python main.py --mode download --query "cat:cs.CL AND Transformer" --max-results 100
    
    # Train classifier
    python main.py --mode train --model classifier --epochs 10
    
    # Ingest PDFs
    python main.py --mode ingest --strategy recursive
    
    # Ask a question
    python main.py --mode ask --question "What are the main limitations of transformers?"
    
    # Interactive session
    python main.py --mode interactive --llm flan-t5-base
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_mode(args):
    """Download papers from ArXiv."""
    from src.data_loader import fetch_arxiv_data, fetch_diverse_dataset, get_dataset_stats
    
    print("\n" + "="*60)
    print("📥 DOWNLOADING PAPERS FROM ARXIV")
    print("="*60)
    
    if args.diverse:
        # Distribute across 5 topics for diverse dataset
        papers_per_topic = args.max_results // 5
        papers = fetch_diverse_dataset(
            papers_per_topic=papers_per_topic,
            download_pdfs=False  # First get abstracts only
        )
        # Now download PDFs separately (limited count)
        if args.download_pdfs:
            print(f"\n📄 Downloading {args.num_pdfs} PDFs for RAG...")
            fetch_arxiv_data(
                query=args.query,
                max_results=args.num_pdfs,
                download_pdfs=True
            )
    else:
        # Download abstracts (large count)
        papers = fetch_arxiv_data(
            query=args.query,
            max_results=args.max_results,
            download_pdfs=False
        )
        # Download PDFs separately (limited count)
        if args.download_pdfs:
            print(f"\n📄 Downloading {args.num_pdfs} PDFs for RAG...")
            fetch_arxiv_data(
                query=args.query,
                max_results=args.num_pdfs,
                download_pdfs=True,
                append_mode=True
            )
    
    stats = get_dataset_stats()
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def train_mode(args):
    """Train Phase 1 transformer models."""
    from src.phase1_transformers.train import (
        train_classifier, 
        train_language_model, 
        train_seq2seq,
        train_all_models,
        load_papers
    )
    
    print("\n" + "="*60)
    print("🎓 PHASE 1: TRAINING TRANSFORMER MODELS")
    print("="*60)
    
    # Load papers
    papers = load_papers()
    print(f"Loaded {len(papers)} papers for training")
    
    if args.model == 'classifier':
        train_classifier(
            papers=papers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.model == 'lm':
        train_language_model(
            papers=papers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.model == 'seq2seq':
        train_seq2seq(
            papers=papers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    elif args.model == 'all':
        train_all_models(papers)
    
    print("\n✅ Training complete!")
    print(f"   Checkpoints saved to: {PROJECT_ROOT / 'models' / 'phase1_checkpoints'}")
    print(f"   Logs saved to: {PROJECT_ROOT / 'logs'}")


def ingest_mode(args):
    """Ingest PDFs for RAG pipeline."""
    from src.phase2_rag.ingestion import ingest_pdfs, compare_chunking_strategies
    
    print("\n" + "="*60)
    print("📄 PHASE 2: INGESTING PDFs FOR RAG")
    print("="*60)
    
    if args.compare:
        print("\n📊 Comparing chunking strategies...")
        results = compare_chunking_strategies()
        for strategy, stats in results.items():
            print(f"\n{strategy}:")
            for key, value in stats.items():
                print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
    else:
        ingestor = ingest_pdfs(
            pdf_dir=args.pdf_dir,
            index_path=args.index_path,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_hybrid=args.hybrid
        )
        
        print("\n✅ Ingestion complete!")
        print(f"   Vector store saved to: {PROJECT_ROOT / 'data' / 'vector_store'}")


def ask_mode(args):
    """Ask a question using RAG."""
    from src.phase2_rag.rag_engine import RAGEngine
    
    print("\n" + "="*60)
    print("❓ PHASE 2: RAG Q&A")
    print("="*60)
    
    # Initialize engine
    engine = RAGEngine(
        model_name=args.llm,
        vector_store_path=args.index_path,
        use_quantization=args.quantize
    )
    engine.initialize()
    
    # Answer question
    print(f"\n🔍 Question: {args.question}")
    print("\n⏳ Searching and generating answer...")
    
    result = engine.generate_answer(args.question)
    
    print(f"\n💡 Answer:\n{result.answer}")
    
    if result.citations:
        print("\n📚 Citations:")
        for c in result.citations:
            print(f"   - {c['paper_id']} [{c['section']}]")
    
    print(f"\n📊 Confidence: {result.confidence:.2%}")
    
    if args.show_sources:
        print("\n📄 Source Documents:")
        for i, doc in enumerate(result.source_documents[:3]):
            print(f"\n   [{i+1}] Paper: {doc.paper_id}, Section: {doc.section}")
            print(f"       {doc.text[:200]}...")


def interactive_mode(args):
    """Run interactive Q&A session."""
    from src.phase2_rag.rag_engine import RAGEngine, interactive_qa
    
    print("\n" + "="*60)
    print("🔬 INTERACTIVE Q&A SESSION")
    print("="*60)
    
    # Initialize engine
    engine = RAGEngine(
        model_name=args.llm,
        vector_store_path=args.index_path,
        use_quantization=args.quantize
    )
    engine.initialize()
    
    # Run interactive session
    interactive_qa(engine)


def evaluate_mode(args):
    """Run evaluation metrics."""
    from src.phase2_rag.rag_engine import RAGEngine
    import json
    
    print("\n" + "="*60)
    print("📊 EVALUATION MODE")
    print("="*60)
    
    # Initialize engine
    engine = RAGEngine(
        model_name=args.llm,
        vector_store_path=args.index_path,
        use_quantization=args.quantize
    )
    engine.initialize()
    
    # Sample evaluation queries
    eval_queries = [
        {"query": "What are the main contributions of transformer models?", "relevant_papers": []},
        {"query": "How does attention mechanism work?", "relevant_papers": []},
        {"query": "What are the limitations of current NLP models?", "relevant_papers": []},
        {"query": "How to improve retrieval augmented generation?", "relevant_papers": []},
        {"query": "What baselines are commonly used in NLP?", "relevant_papers": []},
    ]
    
    if args.queries_file:
        with open(args.queries_file, 'r') as f:
            eval_queries = json.load(f)
    
    print(f"\n📝 Evaluating with {len(eval_queries)} queries...")
    
    # Retrieval evaluation
    retrieval_metrics = engine.evaluate_retrieval(
        eval_queries,
        k_values=[1, 3, 5, 10]
    )
    
    print("\n📊 Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Faithfulness check on sample
    print("\n📝 Faithfulness Check (sample):")
    for query_data in eval_queries[:3]:
        answer = engine.generate_answer(query_data['query'])
        faithfulness = engine.check_faithfulness(answer)
        print(f"\n   Query: {query_data['query'][:50]}...")
        print(f"   Faithfulness: {faithfulness['faithfulness_score']:.2%}")
        print(f"   Has Citations: {faithfulness['has_citations']}")


def related_work_mode(args):
    """Generate related work paragraph."""
    from src.phase2_rag.rag_engine import RAGEngine
    
    print("\n" + "="*60)
    print("📝 RELATED WORK GENERATION")
    print("="*60)
    
    # Initialize engine
    engine = RAGEngine(
        model_name=args.llm,
        vector_store_path=args.index_path,
        use_quantization=args.quantize
    )
    engine.initialize()
    
    print(f"\n📌 Topic: {args.topic}")
    print("\n⏳ Generating related work paragraph...")
    
    result = engine.generate_related_work(args.topic)
    
    print(f"\n📝 Related Work:\n{result.paragraph}")
    print(f"\n📚 Cited Papers: {', '.join(result.cited_papers)}")


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper Companion AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download papers:      python main.py --mode download --query "Transformer" --max-results 50
  Train all models:     python main.py --mode train --model all --epochs 10
  Ingest PDFs:          python main.py --mode ingest --strategy recursive
  Ask question:         python main.py --mode ask --question "What is attention?"
  Interactive mode:     python main.py --mode interactive
  Generate related work: python main.py --mode related --topic "language models"
        """
    )
    
    # Main mode argument
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['download', 'train', 'ingest', 'ask', 'interactive', 'evaluate', 'related', 'compare'],
        help="Operation mode"
    )
    
    # Download arguments
    parser.add_argument('--query', type=str, default="cat:cs.CL AND (LLM OR Transformer)",
                        help="ArXiv search query")
    parser.add_argument('--max-results', type=int, default=10000,
                        help="Maximum papers to download (default: 10000 for abstracts)")
    parser.add_argument('--num-pdfs', type=int, default=500,
                        help="Number of PDFs to download for RAG")
    parser.add_argument('--download-pdfs', action='store_true',
                        help="Also download PDFs")
    parser.add_argument('--diverse', action='store_true',
                        help="Download diverse dataset across topics")
    
    # Training arguments
    parser.add_argument('--model', type=str, default='all',
                        choices=['classifier', 'lm', 'seq2seq', 'all'],
                        help="Model to train")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Training batch size")
    parser.add_argument('--d-model', type=int, default=256,
                        help="Model dimension")
    parser.add_argument('--num-heads', type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument('--num-layers', type=int, default=4,
                        help="Number of transformer layers")
    
    # Ingestion arguments
    parser.add_argument('--pdf-dir', type=str, default=None,
                        help="Directory containing PDFs")
    parser.add_argument('--index-path', type=str, default=None,
                        help="Path to save/load vector store")
    parser.add_argument('--strategy', type=str, default='recursive',
                        choices=['fixed_size', 'section_based', 'recursive', 'hybrid'],
                        help="Chunking strategy")
    parser.add_argument('--chunk-size', type=int, default=500,
                        help="Size of text chunks")
    parser.add_argument('--chunk-overlap', type=int, default=50,
                        help="Overlap between chunks")
    parser.add_argument('--hybrid', action='store_true',
                        help="Use hybrid retrieval (BM25 + dense)")
    parser.add_argument('--compare', action='store_true',
                        help="Compare chunking strategies")
    
    # RAG arguments
    parser.add_argument('--question', type=str,
                        help="Question to ask")
    parser.add_argument('--topic', type=str,
                        help="Topic for related work generation")
    parser.add_argument('--llm', type=str, default='flan-t5-base',
                        help="LLM to use for generation")
    parser.add_argument('--quantize', action='store_true',
                        help="Use 4-bit quantization")
    parser.add_argument('--show-sources', action='store_true',
                        help="Show source documents in answer")
    
    # Evaluation arguments
    parser.add_argument('--queries-file', type=str, default=None,
                        help="JSON file with evaluation queries")
    
    args = parser.parse_args()
    
    # Route to appropriate handler
    try:
        if args.mode == 'download':
            download_mode(args)
        elif args.mode == 'train':
            train_mode(args)
        elif args.mode == 'ingest':
            ingest_mode(args)
        elif args.mode == 'ask':
            if not args.question:
                print("Error: --question is required for ask mode")
                sys.exit(1)
            ask_mode(args)
        elif args.mode == 'interactive':
            interactive_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
        elif args.mode == 'related':
            if not args.topic:
                print("Error: --topic is required for related mode")
                sys.exit(1)
            related_work_mode(args)
        elif args.mode == 'compare':
            args.compare = True
            ingest_mode(args)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you have downloaded data first with --mode download")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
