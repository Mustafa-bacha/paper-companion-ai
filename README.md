# Research Paper Companion AI

A comprehensive research assistant that indexes open-access papers and answers technical questions with citations. It also generates related work paragraphs grounded strictly in retrieved passages.

## 🎯 Project Overview

This project implements a complete research assistant with two main phases:

### Phase 1: Transformer Variants from Scratch
- **Encoder-Only (TextClassifier)**: Classifies paper abstracts into topics (NLP/CV/Security/Healthcare/ML)
- **Decoder-Only (DecoderOnlyLM)**: Tiny language model for next-token prediction
- **Encoder-Decoder (EncoderDecoderTransformer)**: Generates TL;DR summaries from abstracts

### Phase 2: RAG Pipeline
- PDF ingestion with section detection
- Multiple chunking strategies (fixed-size, section-based, recursive)
- Vector indexing with FAISS
- Citation-aware answer generation
- Related work paragraph generation
- Evaluation metrics (Recall@k, Faithfulness)

## 📁 Project Structure

```
paper_companion_ai/
├── data/
│   ├── raw_pdfs/             # PDFs downloaded from ArXiv
│   ├── abstracts.jsonl       # Extracted abstracts for Phase 1
│   └── vector_store/         # FAISS index storage
├── logs/                     # TensorBoard logs
├── models/
│   ├── phase1_checkpoints/   # Saved model checkpoints
│   └── tokenizer/            # Custom BPE tokenizer
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # ArXiv API integration
│   ├── phase1_transformers/
│   │   ├── __init__.py
│   │   ├── layers.py         # Attention, FFN, PositionalEncoding
│   │   ├── models.py         # Encoder, Decoder, Seq2Seq models
│   │   └── train.py          # Training pipelines
│   └── phase2_rag/
│       ├── __init__.py
│       ├── ingestion.py      # PDF parsing & chunking
│       └── rag_engine.py     # Retrieval & generation
├── requirements.txt
├── main.py                   # CLI entry point
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

```bash
# Clone or create the project directory
cd paper_companion_ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup (Recommended)
For CUDA acceleration:
```bash
# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# FAISS GPU
pip install faiss-gpu
```

## 📖 Usage

### Quick Start

```bash
# 1. Download papers from ArXiv
python main.py --mode download --query "cat:cs.CL AND Transformer" --max-results 100 --download-pdfs

# 2. Train Phase 1 models
python main.py --mode train --model all --epochs 10

# 3. Ingest PDFs for RAG
python main.py --mode ingest --strategy recursive

# 4. Ask questions
python main.py --mode ask --question "What are the main contributions of transformer models?"

# 5. Interactive session
python main.py --mode interactive
```

### Detailed Commands

#### Download Papers
```bash
# Download with specific query
python main.py --mode download --query "cat:cs.AI AND (RAG OR retrieval)" --max-results 50

# Download diverse dataset (balanced across topics)
python main.py --mode download --diverse --max-results 200

# Also download PDFs for RAG
python main.py --mode download --query "deep learning" --max-results 30 --download-pdfs
```

#### Train Models
```bash
# Train specific model
python main.py --mode train --model classifier --epochs 20 --batch-size 16

# Train with custom architecture
python main.py --mode train --model lm --d-model 256 --num-heads 4 --num-layers 3

# Train all three variants
python main.py --mode train --model all --epochs 10
```

#### Ingest PDFs
```bash
# Basic ingestion
python main.py --mode ingest

# With custom chunking
python main.py --mode ingest --strategy section_based --chunk-size 600

# Compare chunking strategies
python main.py --mode compare
```

#### RAG Q&A
```bash
# Single question
python main.py --mode ask --question "What baselines are used in NLP research?"

# With source display
python main.py --mode ask --question "How does attention work?" --show-sources

# With different LLM
python main.py --mode ask --question "..." --llm phi-2 --quantize

# Interactive mode
python main.py --mode interactive --llm flan-t5-base
```

#### Generate Related Work
```bash
python main.py --mode related --topic "retrieval augmented generation"
```

#### Evaluation
```bash
python main.py --mode evaluate --llm flan-t5-base
```

## 🔧 Configuration

### Model Options

| Model Name | HuggingFace ID | Size | Notes |
|------------|----------------|------|-------|
| `flan-t5-base` | google/flan-t5-base | 250M | Recommended for limited GPU |
| `flan-t5-small` | google/flan-t5-small | 77M | Fastest |
| `phi-2` | microsoft/phi-2 | 2.7B | Best quality, needs quantization |
| `gpt2` | gpt2 | 124M | Decoder-only baseline |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Good balance |

### Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `fixed_size` | Simple character-based chunks | Baseline |
| `section_based` | Respects paper sections | Academic papers |
| `recursive` | Smart splitting at paragraph boundaries | General use |
| `hybrid` | Section-aware with size limits | Recommended |

## 📊 Phase 1: Transformer Implementation

### Architecture Details

All models follow project constraints:
- `d_model`: 128-256
- `num_layers`: 2-4
- `num_heads`: 2-4
- `max_seq_len`: 128-256

### Implemented Components (from scratch)

1. **Positional Encoding**: Sinusoidal position embeddings
2. **Multi-Head Attention**: Scaled dot-product attention with multiple heads
3. **Feed-Forward Network**: Position-wise fully connected layers
4. **Layer Normalization**: Pre-norm configuration
5. **Causal Masking**: For decoder self-attention

### Training Output

```
📊 Training produces:
- Loss curves (TensorBoard)
- Model checkpoints (models/phase1_checkpoints/)
- Evaluation metrics (Accuracy, Perplexity, ROUGE)
- Qualitative samples
```

View TensorBoard logs:
```bash
tensorboard --logdir logs/
```

## 📚 Phase 2: RAG Pipeline

### Features

1. **PDF Processing**
   - Text extraction with PyMuPDF
   - Section detection (Abstract, Methods, Results, etc.)
   - Metadata preservation

2. **Chunking Strategies**
   - Fixed-size chunks
   - Section-based chunks
   - Recursive character splitting
   - Hybrid approach

3. **Retrieval**
   - Dense retrieval (FAISS + sentence-transformers)
   - Optional hybrid retrieval (BM25 + Dense)
   - GPU-accelerated embeddings

4. **Generation**
   - Citation-aware prompts
   - 4-bit quantization support
   - Multiple LLM backends

### Citation Format

Answers include citations in the format:
```
[Paper: 2312.12345, Section: Methods]
```

## 🧪 Evaluation

### Metrics

1. **Retrieval Quality**
   - Recall@1, Recall@3, Recall@5, Recall@10

2. **Generation Quality**
   - Faithfulness score
   - Citation presence
   - Term overlap with sources

3. **Phase 1 Metrics**
   - Classification: Accuracy
   - Language Model: Perplexity
   - Summarization: ROUGE-1, ROUGE-2, ROUGE-L

## 🤖 Supported LLMs

The RAG pipeline supports:
- **Local models**: flan-t5, phi-2, gpt2, TinyLlama
- **Quantization**: 4-bit via bitsandbytes
- **Fine-tuning**: LoRA/QLoRA via PEFT (optional)

## 📝 References

This implementation draws from:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) (Harvard NLP)
- [minGPT](https://github.com/karpathy/minGPT) (Andrej Karpathy)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- ArXiv for providing open access to research papers
- HuggingFace for the Transformers library
- The open-source ML community

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use quantization
   python main.py --mode ask --question "..." --quantize
   
   # Or use smaller model
   python main.py --mode ask --question "..." --llm flan-t5-small
   ```

2. **No PDFs Found**
   ```bash
   # Download PDFs first
   python main.py --mode download --download-pdfs --max-results 30
   ```

3. **Tokenizer Not Found**
   ```bash
   # Train Phase 1 models first (creates tokenizer)
   python main.py --mode train --model classifier --epochs 1
   ```

4. **Import Errors**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --upgrade
   ```
