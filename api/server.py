#!/usr/bin/env python3
"""
FastAPI Backend Server for Research Paper Companion AI

Provides REST API endpoints for:
- Phase 1: Text Classification, Language Generation, Summarization
- Phase 2: RAG Q&A with model selection
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Literal
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import uvicorn

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=10)

class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for text generation", min_length=1)
    max_length: int = Field(100, ge=10, le=500)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=100)

class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to summarize", min_length=50)
    max_length: int = Field(100, ge=20, le=200)

class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int

class RAGRequest(BaseModel):
    question: str = Field(..., description="Question to ask", min_length=5)
    model: Literal["flan-t5-small", "flan-t5-base"] = "flan-t5-small"
    num_sources: int = Field(5, ge=1, le=10)
    show_sources: bool = True

class RAGSource(BaseModel):
    paper_id: str
    section: str
    page: int
    text: str
    score: float

class RAGResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[RAGSource]
    model_used: str

class RelatedWorkRequest(BaseModel):
    topic: str = Field(..., description="Topic for related work", min_length=5)
    model: Literal["flan-t5-small", "flan-t5-base"] = "flan-t5-small"

class RelatedWorkResponse(BaseModel):
    paragraph: str
    cited_papers: List[str]

class HealthResponse(BaseModel):
    status: str
    phase1_loaded: bool
    phase2_loaded: bool
    device: str


# =============================================================================
# GLOBAL STATE
# =============================================================================

class ModelState:
    """Global state for loaded models."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Phase 1 models
        self.classifier = None
        self.language_model = None
        self.seq2seq = None
        self.tokenizer = None
        
        # Phase 2 RAG
        self.rag_engines = {}  # model_name -> RAGEngine
        self.vector_store_loaded = False
        
        # Labels for classifier (will be loaded from checkpoint)
        self.labels = []
        self.label2id = {}

state = ModelState()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_phase1_models():
    """Load Phase 1 transformer models."""
    from src.phase1_transformers.models import TextClassifier, DecoderOnlyLM, EncoderDecoderTransformer
    from tokenizers import Tokenizer
    
    print("🔄 Loading Phase 1 models...")
    
    checkpoint_dir = PROJECT_ROOT / "models" / "phase1_checkpoints"
    tokenizer_path = PROJECT_ROOT / "models" / "tokenizer" / "tokenizer.json"
    
    # Load tokenizer
    if tokenizer_path.exists():
        state.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = state.tokenizer.get_vocab_size()
        print(f"   ✅ Tokenizer loaded (vocab: {vocab_size})")
    else:
        print(f"   ❌ Tokenizer not found at {tokenizer_path}")
        return
    
    # Model config (must match training config!)
    d_model = 256
    num_heads = 4
    num_layers = 4
    max_seq_len_classifier = 256  # Classifier uses 256
    max_seq_len_lm = 128  # LM was trained with 128
    max_seq_len_seq2seq = 256  # Seq2seq uses 256
    
    # Helper to load checkpoint (handles both formats)
    def load_checkpoint(path, extract_labels=False):
        checkpoint = torch.load(path, map_location=state.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if extract_labels and 'label2id' in checkpoint:
                state.label2id = checkpoint['label2id']
                state.labels = list(state.label2id.keys())
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
        return checkpoint
    
    # Load classifier (extract labels from checkpoint)
    classifier_path = checkpoint_dir / "classifier_best.pt"
    if classifier_path.exists():
        # First load checkpoint to get num_classes
        ckpt = torch.load(classifier_path, map_location=state.device, weights_only=False)
        if 'label2id' in ckpt:
            state.label2id = ckpt['label2id']
            state.labels = list(state.label2id.keys())
        num_classes = len(state.labels) if state.labels else 7
        
        state.classifier = TextClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            max_seq_len=max_seq_len_classifier,
            dropout=0.3
        ).to(state.device)
        state.classifier.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        state.classifier.eval()
        print(f"   ✅ Classifier loaded ({num_classes} classes: {state.labels})")
    
    # Load language model
    lm_path = checkpoint_dir / "lm_best.pt"
    if lm_path.exists():
        state.language_model = DecoderOnlyLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len_lm,
            dropout=0.15
        ).to(state.device)
        lm_ckpt = torch.load(lm_path, map_location=state.device, weights_only=False)
        state.language_model.load_state_dict(lm_ckpt['model_state_dict'] if 'model_state_dict' in lm_ckpt else lm_ckpt)
        state.language_model.eval()
        print("   ✅ Language Model loaded")
    
    # Load seq2seq
    seq2seq_path = checkpoint_dir / "seq2seq_best.pt"
    if seq2seq_path.exists():
        state.seq2seq = EncoderDecoderTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            max_seq_len=max_seq_len_seq2seq,
            dropout=0.2
        ).to(state.device)
        seq2seq_ckpt = torch.load(seq2seq_path, map_location=state.device, weights_only=False)
        state.seq2seq.load_state_dict(seq2seq_ckpt['model_state_dict'] if 'model_state_dict' in seq2seq_ckpt else seq2seq_ckpt)
        state.seq2seq.eval()
        print("   ✅ Seq2Seq Model loaded")
    
    print("✅ Phase 1 models ready!")


def load_phase2_rag(model_name: str = "flan-t5-small"):
    """Load Phase 2 RAG engine with specified model."""
    from src.phase2_rag.rag_engine import RAGEngine
    
    if model_name in state.rag_engines:
        return state.rag_engines[model_name]
    
    print(f"🔄 Loading RAG engine with {model_name}...")
    
    engine = RAGEngine(
        model_name=model_name,
        vector_store_path=PROJECT_ROOT / "data" / "vector_store",
        use_quantization=False  # Disable for API stability
    )
    engine.initialize()
    
    state.rag_engines[model_name] = engine
    state.vector_store_loaded = True
    
    print(f"✅ RAG engine ({model_name}) ready!")
    return engine


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def classify_text(text: str) -> ClassifyResponse:
    """Classify text using the encoder transformer."""
    if state.classifier is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")
    
    # Tokenize
    encoding = state.tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids[:256]]).to(state.device)
    
    # Inference
    with torch.no_grad():
        logits = state.classifier(input_ids)
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Get prediction
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    
    probabilities = {label: probs[i].item() for i, label in enumerate(state.labels)}
    
    return ClassifyResponse(
        category=state.labels[pred_idx],
        confidence=confidence,
        probabilities=probabilities
    )


def generate_text(prompt: str, max_length: int, temperature: float, top_k: int) -> GenerateResponse:
    """Generate text using the decoder-only transformer."""
    if state.language_model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Language model not loaded")
    
    # Tokenize prompt
    encoding = state.tokenizer.encode(prompt)
    input_ids = encoding.ids[:128]  # Limit prompt length
    
    generated = list(input_ids)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            x = torch.tensor([generated[-128:]]).to(state.device)  # Use max 128 tokens
            output = state.language_model(x)
            
            # Model returns (logits, loss) tuple
            logits = output[0] if isinstance(output, tuple) else output
            
            # Apply temperature - get last token logits
            logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Sample
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            
            # Check for EOS
            if next_token == state.tokenizer.token_to_id("[EOS]"):
                break
            
            generated.append(next_token)
    
    # Decode
    generated_text = state.tokenizer.decode(generated)
    
    return GenerateResponse(
        generated_text=generated_text,
        tokens_generated=len(generated) - len(input_ids)
    )


def summarize_text(text: str, max_length: int) -> SummarizeResponse:
    """Summarize text using the encoder-decoder transformer."""
    if state.seq2seq is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="Seq2Seq model not loaded")
    
    # Tokenize input
    encoding = state.tokenizer.encode(text)
    src_ids = encoding.ids[:256]
    src = torch.tensor([src_ids]).to(state.device)
    
    # Start with BOS token
    bos_id = state.tokenizer.token_to_id("[BOS]")
    eos_id = state.tokenizer.token_to_id("[EOS]")
    
    generated = [bos_id]
    
    # Generate summary
    with torch.no_grad():
        for _ in range(max_length):
            tgt = torch.tensor([generated]).to(state.device)
            output = state.seq2seq(src, tgt)
            
            # Model returns (logits, loss) tuple
            logits = output[0] if isinstance(output, tuple) else output
            
            next_token = logits[0, -1, :].argmax().item()
            
            if next_token == eos_id:
                break
            
            generated.append(next_token)
    
    # Decode (skip BOS)
    summary = state.tokenizer.decode(generated[1:])
    
    return SummarizeResponse(
        summary=summary,
        original_length=len(text),
        summary_length=len(summary)
    )


def rag_query(request: RAGRequest) -> RAGResponse:
    """Query the RAG system."""
    try:
        engine = load_phase2_rag(request.model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load RAG engine: {str(e)}")
    
    # Get answer
    result = engine.generate_answer(
        question=request.question,
        k=request.num_sources,
        include_sources=request.show_sources
    )
    
    # Format sources
    sources = []
    if request.show_sources:
        for chunk in result.source_documents[:request.num_sources]:
            sources.append(RAGSource(
                paper_id=chunk.paper_id,
                section=chunk.section,
                page=chunk.page,
                text=chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                score=chunk.score
            ))
    
    return RAGResponse(
        answer=result.answer,
        confidence=result.confidence,
        sources=sources,
        model_used=request.model
    )


def generate_related_work(request: RelatedWorkRequest) -> RelatedWorkResponse:
    """Generate related work paragraph."""
    try:
        engine = load_phase2_rag(request.model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to load RAG engine: {str(e)}")
    
    result = engine.generate_related_work(topic=request.topic, k=10)
    
    return RelatedWorkResponse(
        paragraph=result.paragraph,
        cited_papers=result.cited_papers
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("\n" + "="*60)
    print("🚀 Starting Research Paper Companion API")
    print("="*60 + "\n")
    
    # Load Phase 1 models
    try:
        load_phase1_models()
    except Exception as e:
        print(f"⚠️ Failed to load Phase 1 models: {e}")
    
    # Pre-load default RAG model
    try:
        load_phase2_rag("flan-t5-small")
    except Exception as e:
        print(f"⚠️ Failed to load RAG engine: {e}")
    
    print("\n" + "="*60)
    print("✅ API Server Ready!")
    print("="*60 + "\n")
    
    yield
    
    print("\n🛑 Shutting down server...")


app = FastAPI(
    title="Research Paper Companion AI",
    description="API for Phase 1 Transformer models and Phase 2 RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        phase1_loaded=state.classifier is not None,
        phase2_loaded=state.vector_store_loaded,
        device=state.device
    )


@app.get("/api/models")
async def list_models():
    """List available models and their status."""
    return {
        "phase1": {
            "classifier": {
                "loaded": state.classifier is not None,
                "type": "Encoder-only Transformer",
                "description": "Classifies research papers into 5 categories"
            },
            "language_model": {
                "loaded": state.language_model is not None,
                "type": "Decoder-only Transformer",
                "description": "Generates research paper text"
            },
            "seq2seq": {
                "loaded": state.seq2seq is not None,
                "type": "Encoder-Decoder Transformer",
                "description": "Generates TL;DR summaries"
            }
        },
        "phase2": {
            "rag": {
                "loaded": state.vector_store_loaded,
                "available_models": ["flan-t5-small", "flan-t5-base"],
                "loaded_models": list(state.rag_engines.keys()),
                "description": "RAG-based Q&A with citations"
            }
        }
    }


# Phase 1 Endpoints
@app.post("/api/phase1/classify", response_model=ClassifyResponse)
async def api_classify(request: ClassifyRequest):
    """Classify text into research categories."""
    return classify_text(request.text)


@app.post("/api/phase1/generate", response_model=GenerateResponse)
async def api_generate(request: GenerateRequest):
    """Generate text continuation."""
    return generate_text(
        request.prompt,
        request.max_length,
        request.temperature,
        request.top_k
    )


@app.post("/api/phase1/summarize", response_model=SummarizeResponse)
async def api_summarize(request: SummarizeRequest):
    """Generate TL;DR summary."""
    return summarize_text(request.text, request.max_length)


# Phase 2 Endpoints
@app.post("/api/phase2/ask", response_model=RAGResponse)
async def api_rag_ask(request: RAGRequest):
    """Ask a question using RAG."""
    return rag_query(request)


@app.post("/api/phase2/related-work", response_model=RelatedWorkResponse)
async def api_related_work(request: RelatedWorkRequest):
    """Generate related work paragraph."""
    return generate_related_work(request)


@app.get("/api/phase2/stats")
async def api_rag_stats():
    """Get RAG system statistics."""
    if not state.vector_store_loaded:
        raise HTTPException(status_code=503, detail="RAG not loaded")
    
    engine = list(state.rag_engines.values())[0] if state.rag_engines else None
    
    if engine and hasattr(engine, 'vector_store') and engine.vector_store:
        return {
            "status": "loaded",
            "total_documents": engine.vector_store.index.ntotal if hasattr(engine.vector_store, 'index') else "unknown",
            "embedding_model": engine.embedding_model,
            "loaded_llms": list(state.rag_engines.keys())
        }
    
    return {"status": "partially_loaded"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)]
    )
