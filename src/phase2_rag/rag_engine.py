"""
RAG Engine Module - Retrieval Augmented Generation Pipeline

This module implements:
1. Vector retrieval from FAISS
2. LLM integration with quantization (4-bit)
3. Citation-aware answer generation
4. Related work paragraph generation
5. Evaluation metrics (Recall@k, Faithfulness)

Supported LLM backends:
- HuggingFace local models (phi-2, flan-t5, gpt2, etc.)
- 4-bit quantized models via bitsandbytes
- Optional LoRA/QLoRA fine-tuning via PEFT
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
import re

import torch
from tqdm import tqdm

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    pipeline
)

# LangChain
try:
    from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.chains import RetrievalQA
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain components not fully available: {e}")

# PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk."""
    text: str
    paper_id: str
    section: str
    page: int
    score: float
    chunk_id: str


@dataclass
class CitedAnswer:
    """Represents an answer with citations."""
    answer: str
    citations: List[Dict]
    source_documents: List[RetrievedChunk]
    confidence: float


@dataclass
class RelatedWorkParagraph:
    """Represents a generated related work paragraph."""
    paragraph: str
    cited_papers: List[str]
    source_chunks: List[RetrievedChunk]


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

QA_PROMPT_TEMPLATE = """You are a precise research assistant. Use ONLY the provided Context to answer the Question.
For every fact you state, cite the source using the format [Paper: <paper_id>, Section: <section>].
If the answer is not found in the context, respond with "I cannot find this information in the provided documents."

Context:
{context}

Question: {question}

Answer (with citations):"""


RELATED_WORK_PROMPT_TEMPLATE = """You are an academic writer generating a related work paragraph.
Based on the following research paper excerpts, write a coherent paragraph discussing related work on the topic: "{topic}".
- Cite papers using the format [Paper ID] after each claim.
- Include 3-5 citations from the provided excerpts.
- Write in formal academic style.
- Only use information from the provided excerpts.

Paper Excerpts:
{context}

Write a related work paragraph (150-250 words):"""


SUMMARY_PROMPT_TEMPLATE = """Based on the following paper sections, provide a concise summary of the main contributions and findings.
Cite specific sections using [Section: <name>].

Paper Content:
{context}

Summary:"""


# =============================================================================
# RAG ENGINE
# =============================================================================

class RAGEngine:
    """
    Complete RAG Pipeline for Research Paper Q&A.
    
    Features:
    - Load and query vector store
    - LLM with 4-bit quantization
    - Citation-aware responses
    - Related work generation
    - Evaluation metrics
    """
    
    SUPPORTED_MODELS = {
        'phi-2': 'microsoft/phi-2',
        'phi-3': 'microsoft/Phi-3-mini-4k-instruct',
        'flan-t5-base': 'google/flan-t5-base',
        'flan-t5-small': 'google/flan-t5-small',
        't5-small': 't5-small',
        'gpt2': 'gpt2',
        'distilgpt2': 'distilgpt2',
        'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    }
    
    def __init__(
        self,
        model_name: str = "flan-t5-base",
        vector_store_path: Path = None,
        use_quantization: bool = True,
        use_lora: bool = False,
        device: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG Engine.
        
        Args:
            model_name: Name or path of LLM to use
            vector_store_path: Path to FAISS index
            use_quantization: Whether to use 4-bit quantization
            use_lora: Whether to use LoRA adapters
            device: Device to use ('cuda', 'cpu', or None for auto)
            embedding_model: Model for embeddings
        """
        self.model_name = model_name
        self.vector_store_path = vector_store_path or VECTOR_STORE_DIR
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        self.embedding_model = embedding_model
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.llm_pipeline = None
        self.vector_store = None
        self.embeddings = None
        self.qa_chain = None
        
        # Tracking
        self.query_history = []
    
    def _get_model_id(self, model_name: str) -> str:
        """Get HuggingFace model ID from name."""
        if model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_name]
        return model_name
    
    def _is_seq2seq(self, model_id: str) -> bool:
        """Check if model is sequence-to-sequence."""
        seq2seq_indicators = ['t5', 'bart', 'pegasus', 'marian']
        return any(ind in model_id.lower() for ind in seq2seq_indicators)
    
    def load_model(self):
        """Load the LLM with optional quantization."""
        model_id = self._get_model_id(self.model_name)
        print(f"🤖 Loading model: {model_id}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quantization_config = None
        if self.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("   Using 4-bit quantization")
        
        # Load model
        is_seq2seq = self._is_seq2seq(model_id)
        model_class = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
        
        load_kwargs = {
            "trust_remote_code": True,
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto" if self.device == "cuda" else None
        
        self.model = model_class.from_pretrained(model_id, **load_kwargs)
        
        # Apply LoRA if requested
        if self.use_lora and PEFT_AVAILABLE:
            print("   Applying LoRA adapters")
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"] if not is_seq2seq else ["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Move to device if not using device_map
        if not quantization_config and self.device != "auto":
            self.model = self.model.to(self.device)
        
        # Create pipeline
        task = "text2text-generation" if is_seq2seq else "text-generation"
        
        pipe_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": 256,
            "temperature": 0.1,
            "do_sample": True,
            "repetition_penalty": 1.15,
        }
        
        if not is_seq2seq:
            pipe_kwargs["return_full_text"] = False
        
        self.llm_pipeline = pipeline(task, **pipe_kwargs)
        print(f"✅ Model loaded successfully")
    
    def load_vector_store(self, path: Path = None):
        """Load the FAISS vector store."""
        path = path or self.vector_store_path
        print(f"📚 Loading vector store from: {path}")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': self.device}
        )
        
        # Load vector store
        self.vector_store = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Vector store loaded")
    
    def initialize(self):
        """Initialize all components."""
        self.load_model()
        self.load_vector_store()
        self._setup_qa_chain()
        print("\n🚀 RAG Engine initialized and ready!")
    
    def _setup_qa_chain(self):
        """Set up the QA chain with LangChain."""
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available. Using direct generation.")
            return
        
        # Wrap pipeline for LangChain
        llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
        
        # Create prompt
        prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def retrieve(
        self,
        query: str,
        k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
        
        Returns:
            List of RetrievedChunk objects
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not loaded. Call initialize() first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        chunks = []
        for doc, score in results:
            chunk = RetrievedChunk(
                text=doc.page_content,
                paper_id=doc.metadata.get('paper_id', 'unknown'),
                section=doc.metadata.get('section', 'unknown'),
                page=doc.metadata.get('page', 0),
                score=float(score),
                chunk_id=doc.metadata.get('chunk_id', '')
            )
            chunks.append(chunk)
        
        return chunks
    
    def generate_answer(
        self,
        question: str,
        k: int = 5,
        include_sources: bool = True
    ) -> CitedAnswer:
        """
        Generate an answer with citations.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            include_sources: Whether to include source documents
        
        Returns:
            CitedAnswer object
        """
        # Retrieve context
        retrieved_chunks = self.retrieve(question, k=k)
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[{i+1}] Paper: {chunk.paper_id}, Section: {chunk.section}\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        
        result = self.llm_pipeline(prompt)[0]
        answer = result.get('generated_text', result.get('text', ''))
        
        # Extract citations from answer
        citations = self._extract_citations(answer, retrieved_chunks)
        
        # Track query
        self.query_history.append({
            'question': question,
            'answer': answer,
            'num_sources': len(retrieved_chunks)
        })
        
        return CitedAnswer(
            answer=answer,
            citations=citations,
            source_documents=retrieved_chunks if include_sources else [],
            confidence=self._estimate_confidence(answer, retrieved_chunks)
        )
    
    def _extract_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> List[Dict]:
        """Extract citation references from the answer."""
        citations = []
        
        # Look for [Paper: X] patterns
        paper_pattern = r'\[Paper:\s*([^\]]+)\]'
        matches = re.findall(paper_pattern, answer)
        
        for match in matches:
            paper_id = match.strip()
            for chunk in chunks:
                if paper_id in chunk.paper_id or chunk.paper_id in paper_id:
                    citations.append({
                        'paper_id': chunk.paper_id,
                        'section': chunk.section,
                        'page': chunk.page
                    })
                    break
        
        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            key = (c['paper_id'], c['section'])
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)
        
        return unique_citations
    
    def _estimate_confidence(
        self,
        answer: str,
        chunks: List[RetrievedChunk]
    ) -> float:
        """Estimate confidence based on retrieval scores and answer quality."""
        if not chunks:
            return 0.0
        
        # Base confidence on retrieval scores
        avg_score = sum(c.score for c in chunks) / len(chunks)
        
        # Lower is better for distance-based scores
        # Convert to 0-1 confidence
        confidence = max(0, 1 - avg_score / 2)
        
        # Penalize if answer says "cannot find" or "don't know"
        negative_phrases = ["cannot find", "don't know", "not mentioned", "not in the context"]
        for phrase in negative_phrases:
            if phrase.lower() in answer.lower():
                confidence *= 0.5
                break
        
        return min(confidence, 1.0)
    
    def generate_related_work(
        self,
        topic: str,
        k: int = 10
    ) -> RelatedWorkParagraph:
        """
        Generate a related work paragraph on a topic.
        
        Args:
            topic: Topic for related work
            k: Number of chunks to retrieve
        
        Returns:
            RelatedWorkParagraph object
        """
        # Retrieve relevant chunks
        query = f"research related to {topic}"
        retrieved_chunks = self.retrieve(query, k=k)
        
        # Format context
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(
                f"[{chunk.paper_id}] ({chunk.section}): {chunk.text[:300]}..."
            )
        context = "\n\n".join(context_parts)
        
        # Generate paragraph
        prompt = RELATED_WORK_PROMPT_TEMPLATE.format(topic=topic, context=context)
        
        result = self.llm_pipeline(prompt)[0]
        paragraph = result.get('generated_text', result.get('text', ''))
        
        # Extract cited papers
        cited_papers = list(set(chunk.paper_id for chunk in retrieved_chunks[:5]))
        
        return RelatedWorkParagraph(
            paragraph=paragraph,
            cited_papers=cited_papers,
            source_chunks=retrieved_chunks
        )
    
    def ask(self, question: str) -> str:
        """
        Simple Q&A interface.
        
        Args:
            question: User question
        
        Returns:
            Answer string
        """
        if self.qa_chain:
            result = self.qa_chain.invoke(question)
            answer = result.get('result', '')
            
            # Add sources
            sources = result.get('source_documents', [])
            if sources:
                answer += "\n\n📚 Sources:\n"
                for doc in sources[:3]:
                    paper_id = doc.metadata.get('paper_id', 'unknown')
                    section = doc.metadata.get('section', 'unknown')
                    answer += f"  - {paper_id} [{section}]\n"
            
            return answer
        else:
            result = self.generate_answer(question)
            return result.answer
    
    def evaluate_retrieval(
        self,
        queries: List[Dict],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            queries: List of {"query": str, "relevant_papers": List[str]}
            k_values: Values of k for Recall@k
        
        Returns:
            Evaluation metrics
        """
        results = {f"recall@{k}": [] for k in k_values}
        
        for query_data in tqdm(queries, desc="Evaluating"):
            query = query_data['query']
            relevant = set(query_data.get('relevant_papers', []))
            
            if not relevant:
                continue
            
            for k in k_values:
                retrieved = self.retrieve(query, k=k)
                retrieved_papers = set(c.paper_id for c in retrieved)
                
                recall = len(retrieved_papers & relevant) / len(relevant)
                results[f"recall@{k}"].append(recall)
        
        # Average
        return {
            metric: sum(values) / len(values) if values else 0
            for metric, values in results.items()
        }
    
    def check_faithfulness(
        self,
        answer: CitedAnswer
    ) -> Dict:
        """
        Check if answer is faithful to source documents.
        
        Simple heuristic check based on:
        - Citation presence
        - Key term overlap with sources
        
        Returns:
            Faithfulness metrics
        """
        if not answer.source_documents:
            return {'faithfulness_score': 0, 'has_citations': False}
        
        # Check for citations
        has_citations = len(answer.citations) > 0
        
        # Check term overlap
        answer_terms = set(answer.answer.lower().split())
        source_terms = set()
        for chunk in answer.source_documents:
            source_terms.update(chunk.text.lower().split())
        
        overlap = len(answer_terms & source_terms) / len(answer_terms) if answer_terms else 0
        
        # Combined score
        faithfulness = 0.5 * float(has_citations) + 0.5 * min(overlap, 1.0)
        
        return {
            'faithfulness_score': faithfulness,
            'has_citations': has_citations,
            'term_overlap': overlap,
            'num_citations': len(answer.citations)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_rag_pipeline(
    index_path: Path = None,
    model_id: str = "google/flan-t5-base",
    use_quantization: bool = True
) -> RAGEngine:
    """
    Convenience function to load a ready-to-use RAG pipeline.
    
    Args:
        index_path: Path to vector store
        model_id: Model to use
        use_quantization: Whether to use 4-bit quantization
    
    Returns:
        Initialized RAGEngine
    """
    engine = RAGEngine(
        model_name=model_id,
        vector_store_path=index_path,
        use_quantization=use_quantization
    )
    engine.initialize()
    return engine


def interactive_qa(engine: RAGEngine):
    """
    Run interactive Q&A session.
    
    Args:
        engine: Initialized RAGEngine
    """
    print("\n" + "="*60)
    print("🔬 Research Paper Companion - Interactive Q&A")
    print("="*60)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - 'related: <topic>' to generate related work")
    print("  - 'exit' or 'quit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not query:
            continue
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if query.lower().startswith('related:'):
            topic = query[8:].strip()
            print("\n⏳ Generating related work paragraph...")
            result = engine.generate_related_work(topic)
            print(f"\n📝 Related Work:\n{result.paragraph}")
            print(f"\n📚 Cited Papers: {', '.join(result.cited_papers)}")
        else:
            print("\n⏳ Searching and generating answer...")
            result = engine.generate_answer(query)
            print(f"\n💡 Answer:\n{result.answer}")
            
            if result.citations:
                print("\n📚 Citations:")
                for c in result.citations:
                    print(f"  - {c['paper_id']} [{c['section']}]")
            
            print(f"\n📊 Confidence: {result.confidence:.2%}")
        
        print("\n" + "-"*40 + "\n")
    
    print("\nGoodbye! 👋")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Engine")
    parser.add_argument('--model', type=str, default='flan-t5-base',
                        help="Model name (e.g., phi-2, flan-t5-base, gpt2)")
    parser.add_argument('--index', type=str, default=None,
                        help="Path to vector store")
    parser.add_argument('--quantize', action='store_true',
                        help="Use 4-bit quantization")
    parser.add_argument('--question', type=str, default=None,
                        help="Single question to ask")
    parser.add_argument('--interactive', action='store_true',
                        help="Run interactive session")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RAGEngine(
        model_name=args.model,
        vector_store_path=args.index,
        use_quantization=args.quantize
    )
    engine.initialize()
    
    if args.question:
        # Answer single question
        result = engine.generate_answer(args.question)
        print(f"\n💡 Answer: {result.answer}")
        print(f"\n📚 Citations: {result.citations}")
    elif args.interactive:
        # Interactive mode
        interactive_qa(engine)
    else:
        # Default: interactive mode
        interactive_qa(engine)
