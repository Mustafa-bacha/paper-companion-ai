# Research Paper Companion AI - Web Interface

Professional web interface for the Research Paper Companion AI system.

## Features

### Phase 1: Transformer Playground
- **Text Classifier**: Encoder-only transformer for paper categorization (CV, ML, NLP, Accel, WS)
- **Language Model**: Decoder-only transformer for text generation
- **Summarizer**: Encoder-decoder transformer for TL;DR summaries

### Phase 2: RAG Q&A System
- Ask questions about AI research
- Get answers with citations from 500+ papers
- Choose between flan-t5-small and flan-t5-base models
- View source documents with relevance scores

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Installation

1. **Install backend dependencies** (if not already done):
```bash
cd paper_companion_ai
pip install fastapi uvicorn
```

2. **Install frontend dependencies**:
```bash
cd web
npm install
```

### Running the Application

**Option 1: Using the start script**
```bash
chmod +x scripts/start_web.sh
./scripts/start_web.sh
```

**Option 2: Manual start**

Terminal 1 - Backend:
```bash
cd paper_companion_ai/api
python server.py
```

Terminal 2 - Frontend:
```bash
cd paper_companion_ai/web
npm run dev
```

### Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                          │
│                   (localhost:3000)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Home     │  │   Phase 1   │  │      Phase 2        │  │
│  │   Landing   │  │  Playground │  │     RAG Q&A         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│                   (localhost:8000)                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────────────────┐  │
│  │    Phase 1 APIs     │  │       Phase 2 APIs           │  │
│  │  /api/phase1/*      │  │      /api/phase2/*           │  │
│  ├─────────────────────┤  ├──────────────────────────────┤  │
│  │ • /classify         │  │ • /ask                       │  │
│  │ • /generate         │  │ • /related-work              │  │
│  │ • /summarize        │  │ • /stats                     │  │
│  └─────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                              │
├─────────────────────────────────────────────────────────────┤
│  Phase 1 Models:              │  Phase 2 RAG:               │
│  • classifier_best.pt        │  • FAISS Vector Store       │
│  • lm_best.pt                │  • 55,986 chunks            │
│  • seq2seq_best.pt           │  • flan-t5-small/base       │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Health Check
- `GET /health` - Check API status and model availability

### Phase 1 - Transformer Models
- `POST /api/phase1/classify` - Classify text into research categories
- `POST /api/phase1/generate` - Generate text continuation
- `POST /api/phase1/summarize` - Generate TL;DR summary

### Phase 2 - RAG System
- `POST /api/phase2/ask` - Ask a question with RAG
- `POST /api/phase2/related-work` - Generate related work paragraph
- `GET /api/phase2/stats` - Get RAG system statistics

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- React 18
- TypeScript
- Tailwind CSS
- Framer Motion (animations)
- Lucide React (icons)

### Backend
- FastAPI
- PyTorch
- HuggingFace Transformers
- LangChain
- FAISS (vector search)

## Development

### Frontend Development
```bash
cd web
npm run dev     # Development mode
npm run build   # Production build
npm run lint    # Lint code
```

### Backend Development
```bash
cd api
python server.py  # Runs with auto-reload
```

## Troubleshooting

### API Connection Issues
- Ensure the backend is running on port 8000
- Check CORS settings if accessing from different origin
- Verify model checkpoints exist in `models/phase1_checkpoints/`

### Model Loading Errors
- Ensure all Phase 1 models are trained
- Check that tokenizer exists in `models/tokenizer/`
- Verify FAISS index exists in `data/vector_store/`

### Frontend Issues
- Clear `.next` cache: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`
