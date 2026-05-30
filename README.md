# Woodmaster CNC Assistant

This project runs a Woodmaster CNC sales and support assistant powered by:

- `data/knowledge.md` as the FAQ knowledge source
- Sarvam for grounded response generation
- session-based memory for short follow-up context
- WhatsApp and Messenger webhook integrations

## Key Files

- `src/config.py`: runtime settings
- `src/rag/prompt.py`: chatbot prompts and guardrails
- `src/rag/generation.py`: Sarvam client wrapper
- `src/pipeline/rag_pipeline.py`: markdown FAQ retrieval and response pipeline
- `src/memory/memory_manager.py`: session history storage
- `src/messenger/conversation_flow.py`: greeting and language flow
- `src/whatsapp/client.py`: WhatsApp send helpers
- `qa.py`: local test script

## Required Environment Variables

```env
SARVAM_API_KEY=your_sarvam_key
SARVAM_GENERATION_MODEL=sarvam-m
SARVAM_TEMPERATURE=0.3
SARVAM_TOP_P=0.85
SARVAM_MAX_OUTPUT_TOKENS=220

WHATSAPP_VERIFY_TOKEN=your_verify_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_ACCESS_TOKEN=your_whatsapp_access_token
WHATSAPP_APP_SECRET=your_whatsapp_app_secret

MESSENGER_ACCESS_TOKEN=your_messenger_access_token
MESSENGER_VERIFY_TOKEN=your_messenger_verify_token

PUBLIC_BASE_URL=https://your-public-domain
SESSION_ENCRYPTION_KEY=optional_custom_key
RAG_TOP_K=5
RAG_MEMORY_TURNS=8
RAG_MAX_CONTEXT_CHARS=5000
```

## Install

```bash
pip install -r requirements.txt
```

## Local QA Test

Edit the question in `qa.py`, then run:

```bash
python qa.py
```

## Webhooks

- `GET /whatsapp/webhook`: Meta verification
- `POST /whatsapp/webhook`: incoming WhatsApp messages
- `GET /messenger/webhook`: Meta verification
- `POST /messenger/webhook`: incoming Messenger messages

## Notes

- The bot only answers Woodmaster CNC related queries.
- FAQ facts come from `data/knowledge.md`.
- If a detail is missing from the FAQ, the bot should say that clearly instead of inventing an answer.
