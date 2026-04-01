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

# WhatsApp Cloud API (Meta)
WHATSAPP_VERIFY_TOKEN=your_random_verify_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_ACCESS_TOKEN=your_permanent_or_temp_access_token
WHATSAPP_GRAPH_VERSION=v23.0
# Optional but recommended for webhook signature validation
WHATSAPP_APP_SECRET=your_meta_app_secret

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

## Real WhatsApp Integration (Meta Cloud API)

This project now supports real WhatsApp webhooks on:

- `GET /whatsapp/webhook` for Meta verification handshake
- `POST /whatsapp/webhook` for incoming WhatsApp messages

### Credentials You Need

1. `WHATSAPP_VERIFY_TOKEN`
2. `WHATSAPP_PHONE_NUMBER_ID`
3. `WHATSAPP_ACCESS_TOKEN`
4. `WHATSAPP_APP_SECRET` (recommended)
5. `GEMINI_API_KEY` (already required by this RAG app)

### How to Get Them (Step by Step)

1. Create a Meta developer app
	- Go to Meta for Developers and create an app (Business type is typical).
2. Add WhatsApp product to the app
	- In your app dashboard, add the WhatsApp product.
3. Get test sender setup
	- In WhatsApp API setup, Meta gives a test phone number and temporary token.
4. Get `WHATSAPP_PHONE_NUMBER_ID`
	- In WhatsApp API setup, copy the Phone Number ID.
5. Generate `WHATSAPP_VERIFY_TOKEN`
	- Create any random long string yourself (for example from a password generator).
	- Put it in `.env` as `WHATSAPP_VERIFY_TOKEN`.
6. Get `WHATSAPP_ACCESS_TOKEN`
	- For testing: use the temporary token shown in Meta dashboard.
	- For production: create a system user token in Meta Business Manager with WhatsApp permissions.
7. Get `WHATSAPP_APP_SECRET` (recommended)
	- In your Meta app settings, copy App Secret and set `WHATSAPP_APP_SECRET`.
8. Configure webhook URL in Meta
	- Public HTTPS URL must point to: `https://<your-domain>/whatsapp/webhook`
	- Verify token must match `WHATSAPP_VERIFY_TOKEN`.
9. Subscribe webhook fields
	- Subscribe at least to `messages` for your WhatsApp business account.
10. Add recipient numbers
	- In test mode, add recipient phone numbers in Meta's allowed recipients list.
11. Send a test message
	- Message your WhatsApp business number from an allowed number.
	- The app reads incoming message, queries RAG, and sends back the answer.

### Production Notes

- Webhook endpoint must be publicly reachable over HTTPS.
- Move from temporary to permanent access token before go-live.
- Keep `WHATSAPP_APP_SECRET` enabled so incoming webhook signatures are validated.
- Configure token rotation and monitoring for reliability.

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

