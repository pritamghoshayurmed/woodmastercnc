# Fast In-Memory RAG (FAISS + Gemini)

This project provides a clean, modular Retrieval-Augmented Generation (RAG) system designed for:

- Fast in-memory retrieval with FAISS
- Multilingual embeddings via Gemini
- Response generation via Gemini
- Multi-turn context and memory management
- Easy integration with WhatsApp and Messenger handlers
- Optional image references in responses

## Architecture

- `src/config.py`: environment and runtime settings
- `src/rag/embeeding.py`: Gemini embedding client
- `src/rag/faiss_store.py`: FAISS vector index build/load/search
- `src/rag/generation.py`: Gemini chat completion client
- `src/memory/memory_manager.py`: per-session in-memory conversation history
- `src/memory/context_manager.py`: retrieved-context assembly and truncation
- `src/pipeline/rag_pipepline.py`: end-to-end RAG pipeline
- `main.py`: local CLI for testing
- `app.py`: simple `ask()` function for integrations

## Environment Variables

Add these values to `.env`:

```env
GEMINI_API_KEY=your_gemini_key

# Optional overrides
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-2.5-flash
GEMINI_TIMEOUT_SECONDS=25
GEMINI_MAX_RETRIES=1
GEMINI_BACKOFF_SECONDS=0.6
GEMINI_TEMPERATURE=0.35
GEMINI_TOP_P=0.9
GEMINI_MAX_OUTPUT_TOKENS=280
GEMINI_THINKING_BUDGET=0

RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=120
RAG_TOP_K=5
RAG_MEMORY_TURNS=8
RAG_MAX_CONTEXT_CHARS=5000
```

## Install

```bash
pip install -r requirements.txt
```

## Run CLI

```bash
python main.py
```

## Integration Usage

```python
from app import ask

response = ask("What machine models are available?", session_id="whatsapp:+91xxxx")
print(response["answer"])
print(response["images"])
```

## Data and Artifacts

- Place text knowledge in `data/*.txt`
- Place product images in `data/images/`
- FAISS index is persisted in `artifacts/faiss.index`
- Chunk metadata is persisted in `artifacts/faiss_meta.json`

## Notes

- The pipeline uses in-memory state for chat memory and loaded index for speed.
- For fresh indexing after data updates, call `initialize(force_rebuild=True)`.
- Image references are auto-selected from retrieved product chunks (for example `product1.png`).

