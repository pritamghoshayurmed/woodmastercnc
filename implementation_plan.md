# WoodMaster CNC — PostgreSQL Persistence & Lead Analytics System
## Implementation Plan

---

## Overview

This plan describes the end-to-end architecture for replacing the current file-based session storage (`artifacts/flow_state/*.json`) with a **Supabase-hosted PostgreSQL database**, and wiring it to the existing FastAPI backend and Next.js dashboard.

The system will:
1. **Identify every incoming user** by phone/PSID, assign them a stable UUID in the `users` table.
2. **Persist full conversation history** (every message with timestamp, sender, language, tokens, model) in the `messages` table.
3. **Track conversation state** (flow stage, current_conversation_id, language, name, address) in the `users` table and the `conversations` table.
4. **Auto-score leads** after every message using the configurable `lead_score_rules` table — results stored in `lead_analysis`.
5. **Generate conversation summaries** via LLM, stored in `conversation_summary`.
6. **Expose REST API endpoints** that the Next.js dashboard consumes for leads, conversations, criteria CRUD.
7. **Be self-hostable** — only a single env var (`DATABASE_URL`) needs to change to move from Supabase to a self-hosted PostgreSQL instance.

---

## Architecture Overview

```
WhatsApp / Messenger / Web Widget
         │
         ▼
  FastAPI server.py
         │
    ┌────┴────────────────┐
    │  DB Layer (new)     │  ← single PostgreSQL URL
    │  src/db/            │
    │  ├── client.py      │  psycopg2 / asyncpg connection pool
    │  ├── models.py      │  dataclass mirrors of DB tables
    │  ├── users.py       │  CRUD for users table
    │  ├── conversations.py│ CRUD for conversations table
    │  ├── messages.py    │  CRUD for messages table
    │  ├── lead_analysis.py│ CRUD for lead_analysis table
    │  ├── lead_score_rules.py  # CRUD for configurable scoring rules
    │  ├── conversation_summary.py
    │  └── events.py      │  optional event log
    └─────────────────────┘
         │
         ▼
  Supabase PostgreSQL (now)  →  Self-hosted PostgreSQL (later, 1 env var)
```

---

## Database Schema

### Table 1 — `users`

Stores one row per unique contact. The `phone_number` is the natural key for WhatsApp/Messenger; web sessions use a synthetic phone-like ID.

```sql
CREATE TABLE users (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number          VARCHAR(20) UNIQUE NOT NULL,
    name                  VARCHAR(100),          -- collected during flow (awaiting_user_info)
    language              VARCHAR(10),           -- selected language code: 'english'|'hindi'|'bengali'
    city                  VARCHAR(100),
    state                 VARCHAR(100),
    country               VARCHAR(100),
    first_seen            TIMESTAMP NOT NULL DEFAULT now(),
    last_seen             TIMESTAMP NOT NULL DEFAULT now(),
    status                VARCHAR(20) NOT NULL DEFAULT 'active', -- active|blocked
    conversation_state    VARCHAR(30) NOT NULL DEFAULT 'LANGUAGE_SELECTION',
    -- Enum: LANGUAGE_SELECTION | ASK_NAME | ASK_LOCATION | FAQ | CONTACT_SHARED | COMPLETED
    current_conversation_id UUID,               -- FK → conversations.id (set after INSERT)
    source                VARCHAR(30) NOT NULL DEFAULT 'WHATSAPP', -- WHATSAPP | MESSENGER | WEB
    created_at            TIMESTAMP NOT NULL DEFAULT now(),
    updated_at            TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_users_phone ON users(phone_number);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_last_seen ON users(last_seen DESC);
```

**conversation_state values** map directly to `ConversationFlowManager` stages:
| DB value | Flow stage |
|---|---|
| `LANGUAGE_SELECTION` | awaiting_language |
| `ASK_NAME` | awaiting_user_info |
| `FAQ` | chatting |
| `CONTACT_SHARED` | contact_forced |
| `COMPLETED` | session ended |

---

### Table 2 — `conversations`

One row per distinct conversation session. A user gets a new conversation when they return after a 15-minute timeout or explicitly restart.

```sql
CREATE TABLE conversations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at        TIMESTAMP NOT NULL DEFAULT now(),
    ended_at          TIMESTAMP,                    -- NULL while active
    status            VARCHAR(20) NOT NULL DEFAULT 'ACTIVE', -- ACTIVE | CLOSED | TRANSFERRED
    source            VARCHAR(30) NOT NULL DEFAULT 'WHATSAPP',
    lead_score        INT NOT NULL DEFAULT 0,
    summary_generated BOOLEAN NOT NULL DEFAULT false,
    contact_shared    BOOLEAN NOT NULL DEFAULT false,
    closed_reason     VARCHAR(200),
    created_at        TIMESTAMP NOT NULL DEFAULT now(),
    updated_at        TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_status ON conversations(status);
CREATE INDEX idx_conversations_started_at ON conversations(started_at DESC);
```

---

### Table 3 — `messages`

Full immutable message log. Every bot reply and user message is appended here.

```sql
CREATE TABLE messages (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender           VARCHAR(10) NOT NULL,     -- 'USER' | 'BOT'
    message_type     VARCHAR(20) NOT NULL DEFAULT 'text', -- text | image | button | file
    message          TEXT NOT NULL,
    language         VARCHAR(10),              -- language at time of message
    timestamp        TIMESTAMP NOT NULL DEFAULT now(),
    tokens           INT,                      -- token count from LLM response
    ai_model         VARCHAR(100),             -- e.g. 'nvidia/nemotron-3-ultra-550b-a55b'
    latency_ms       INT                       -- ms for bot to respond
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX idx_messages_sender ON messages(sender);
```

---

### Table 4 — `lead_analysis`

AI-generated lead qualification per conversation. Updated every N messages (configurable) or on conversation close.

```sql
CREATE TABLE lead_analysis (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id     UUID NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    -- Structured fields from LLM
    intent              VARCHAR(50),       -- e.g. 'purchase', 'inquiry', 'support'
    urgency             VARCHAR(20),       -- low | medium | high
    interest_level      VARCHAR(20),       -- low | medium | high
    product_interest    VARCHAR(200),      -- e.g. '1325 CNC Router, Laser Cutter'
    budget              VARCHAR(100),      -- e.g. 'INR 5,00,000 – 10,00,000'
    timeline            VARCHAR(100),      -- e.g. 'next month', 'Q3 2025'
    sentiment           VARCHAR(20),       -- positive | neutral | negative
    language            VARCHAR(10),
    lead_score          INT NOT NULL DEFAULT 0,    -- 0–100, sum of matched rule weights
    qualified           BOOLEAN NOT NULL DEFAULT false,
    confidence          DECIMAL(5,2),      -- 0.00 – 1.00
    recommended_action  TEXT,             -- LLM-generated next step text
    -- Matched rule IDs (array of lead_score_rules.id values that fired)
    matched_rule_ids    UUID[],
    updated_at          TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_lead_analysis_conversation_id ON lead_analysis(conversation_id);
CREATE INDEX idx_lead_analysis_lead_score ON lead_analysis(lead_score DESC);
CREATE INDEX idx_lead_analysis_qualified ON lead_analysis(qualified);
```

---

### Table 5 — `lead_score_rules`

Admin-configurable scoring rules. The dashboard's **Lead Scoring Criteria** page manages this table via API.

```sql
CREATE TABLE lead_score_rules (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name    VARCHAR(100) NOT NULL,
    description  TEXT,
    weight       INT NOT NULL DEFAULT 10,       -- points added when rule matches
    enabled      BOOLEAN NOT NULL DEFAULT true,
    priority     INT NOT NULL DEFAULT 0,        -- display order
    matching_type VARCHAR(50) NOT NULL DEFAULT 'AI Detected (Recommended)',
    created_at   TIMESTAMP NOT NULL DEFAULT now(),
    updated_at   TIMESTAMP NOT NULL DEFAULT now()
);

-- Seed default rules (mirrors DashboardContext.tsx initialCriteria)
INSERT INTO lead_score_rules (rule_name, description, weight, priority) VALUES
  ('Asked About Price',          'Lead asked about price or quotation',              20, 1),
  ('Product Interest',           'Lead showed interest in specific machine model',   15, 2),
  ('Asked About Features',       'Lead asked about features or specifications',      10, 3),
  ('Timeline Mentioned',         'Lead mentioned purchase timeline',                 15, 4),
  ('Asked for Brochure/Catalog', 'Lead requested brochure or catalog',              10, 5),
  ('Provided Location',          'Lead shared location or city',                     5, 6),
  ('Multiple Interactions',      'Lead came back for another conversation',          10, 7),
  ('Qualified by AI',            'AI marked lead as highly qualified',               25, 8);
```

---

### Table 6 — `conversation_summary`

LLM-generated per-conversation summary, key points, and customer requirements.

```sql
CREATE TABLE conversation_summary (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id      UUID NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    summary              TEXT NOT NULL,
    key_points           JSONB,          -- ["Point 1", "Point 2", ...]
    customer_requirements JSONB,         -- {"machine": "...", "budget": "..."}
    follow_up_needed     BOOLEAN NOT NULL DEFAULT false,
    generated_at         TIMESTAMP NOT NULL DEFAULT now(),
    updated_at           TIMESTAMP NOT NULL DEFAULT now()
);
```

---

### Table 7 — `events` (optional audit/analytics)

Lightweight event log for milestones.

```sql
CREATE TABLE events (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    event_type       VARCHAR(50) NOT NULL,   -- e.g. 'language_selected', 'name_collected', 'lead_scored'
    event_data       JSONB,
    created_at       TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_events_conversation_id ON events(conversation_id);
CREATE INDEX idx_events_type ON events(event_type);
```

---

### Foreign Key Relationship Map

```
users (1) ──────────────────────────── conversations (N)
                                              │
              ┌───────────────────────────────┤
              │                               │
          messages (N)              lead_analysis (1)
                                   conversation_summary (1)
                                   events (N)

lead_score_rules  ←─ read by lead scoring engine ─→ lead_analysis.matched_rule_ids
```

---

## Proposed Changes

### Component 1 — Database Layer (`src/db/`)

#### [NEW] `src/db/__init__.py`
Empty init file.

#### [NEW] `src/db/client.py`
```
- Reads DATABASE_URL from env (Supabase URL or self-hosted postgres URL)
- Creates a connection pool using psycopg2 (sync) or asyncpg (async)
- Exposes get_connection() context manager
- All other DB modules import from here — change just this file to swap databases
```

#### [NEW] `src/db/migrations/001_initial_schema.sql`
```
- CREATE TABLE statements for all 7 tables above
- Seed INSERT for lead_score_rules defaults
- Idempotent (uses IF NOT EXISTS)
```

#### [NEW] `src/db/users.py`
```
- upsert_user(phone_number, source) → User  (INSERT ON CONFLICT DO UPDATE)
- get_user_by_phone(phone_number) → User | None
- update_user_state(user_id, stage, language, name, address)
- update_user_last_seen(user_id)
```

#### [NEW] `src/db/conversations.py`
```
- create_conversation(user_id, source) → Conversation
- get_active_conversation(user_id) → Conversation | None
- close_conversation(conversation_id, reason)
- update_conversation_lead_score(conversation_id, score)
```

#### [NEW] `src/db/messages.py`
```
- append_message(conversation_id, sender, text, language, tokens, model, latency_ms) → int
- get_conversation_messages(conversation_id, limit) → list[Message]
```

#### [NEW] `src/db/lead_analysis.py`
```
- upsert_lead_analysis(conversation_id, analysis_dict) → LeadAnalysis
- get_lead_analysis(conversation_id) → LeadAnalysis | None
```

#### [NEW] `src/db/lead_score_rules.py`
```
- get_enabled_rules() → list[Rule]
- create_rule(name, description, weight, priority) → Rule
- update_rule(rule_id, fields) → Rule
- delete_rule(rule_id) → bool
- reorder_rules(ordered_ids: list[str]) → None
```

#### [NEW] `src/db/conversation_summary.py`
```
- upsert_summary(conversation_id, summary_text, key_points, requirements) → Summary
```

#### [NEW] `src/db/events.py`
```
- log_event(conversation_id, event_type, data) → None
```

---

### Component 2 — Lead Scoring Engine (`src/lead/scorer.py`)

#### [NEW] `src/lead/scorer.py`

Replaces/extends the existing `classifier.py` with **rule-aware scoring**:

```
1. Load enabled rules from lead_score_rules table (cached 60s)
2. For each rule, ask the LLM: "Does this conversation match criterion: <rule_name> — <description>?"
   - Or: run a lightweight keyword/regex check for performance
3. Sum weights of matched rules → lead_score (capped at 100)
4. Determine qualified=True if lead_score >= QUALIFICATION_THRESHOLD (configurable, default 50)
5. Write result to lead_analysis table, update conversations.lead_score
6. Log 'lead_scored' event
```

---

### Component 3 — Conversation Flow Manager (`src/messenger/conversation_flow.py`)

#### [MODIFY] `src/messenger/conversation_flow.py`

The existing file-based `_get_state` / `_save_state` will call the DB layer instead:

```diff
- self._state_dir / f"{safe}.json"   (read/write JSON files)
+ db.users.get_user_by_phone(phone)  (read from PostgreSQL)
+ db.users.update_user_state(...)    (write to PostgreSQL)
```

Key changes:
- On first message: `upsert_user(phone_number, source)` → get/create User row
- On first message: `create_conversation(user_id, source)` → get Conversation row
- Every `_save_state` call: `update_user_state(user_id, stage, language, name, address)`
- Timeout detection: compare `users.last_seen` instead of `state["last_active"]`
- Persist each user message: `append_message(conv_id, 'USER', text, ...)`

The file-based fallback remains active during a `DB_ENABLED=false` dev mode.

---

### Component 4 — RAG Pipeline Hook (`src/pipeline/rag_pipeline.py`)

#### [MODIFY] `src/pipeline/rag_pipeline.py`

After generating a bot reply, record it to DB:

```python
# After LLM call
start = time.time()
response = llm.call(...)
latency_ms = int((time.time() - start) * 1000)

db.messages.append_message(
    conversation_id=...,
    sender='BOT',
    text=response['answer'],
    language=preferred_language,
    tokens=response.get('usage', {}).get('total_tokens'),
    model=settings.llm_model,
    latency_ms=latency_ms,
)

# Trigger async lead scoring (every 3rd bot message, or on conversation close)
if should_score:
    asyncio.create_task(score_lead(conversation_id))
```

---

### Component 5 — Dashboard API (`src/api/`)

A new FastAPI router group providing REST endpoints for the Next.js dashboard to consume **real data** instead of the mock data in `DashboardContext.tsx`.

#### [NEW] `src/api/__init__.py`
#### [NEW] `src/api/router.py`  — registers all sub-routers

#### [NEW] `src/api/leads.py`
```
GET  /api/leads             → paginated list with filters (source, score range, status, language)
GET  /api/leads/{user_id}   → single lead with conversation list
PUT  /api/leads/{user_id}   → update status, assignedTo
GET  /api/leads/export      → CSV export
```

#### [NEW] `src/api/conversations.py`
```
GET  /api/conversations                 → paginated list (date range, status filter)
GET  /api/conversations/{conv_id}       → full conversation with messages + lead_analysis
POST /api/conversations/{conv_id}/close → manually close conversation
```

#### [NEW] `src/api/scoring_rules.py`
```
GET    /api/scoring-rules              → list all rules (ordered by priority)
POST   /api/scoring-rules              → create new rule
PUT    /api/scoring-rules/{rule_id}    → update rule (toggle enabled, change weight)
DELETE /api/scoring-rules/{rule_id}    → delete rule
POST   /api/scoring-rules/reorder      → reorder (drag-and-drop)
```

#### [NEW] `src/api/analytics.py`
```
GET /api/analytics/overview     → total leads, qualified, high-intent, conversations count
GET /api/analytics/by-source    → leads grouped by source
GET /api/analytics/score-dist   → lead score histogram (0-25, 26-50, 51-75, 76-100)
GET /api/analytics/trend        → leads over time (for date range filter)
```

#### [MODIFY] `server.py`
```python
from src.api.router import api_router
app.include_router(api_router, prefix="/api")
```

---

### Component 6 — Configuration (`src/config.py`)

#### [MODIFY] `src/config.py`

Add database settings to the `Settings` dataclass:

```python
@dataclass(frozen=True)
class Settings:
    # ... existing fields ...
    database_url: str          # e.g. postgresql://user:pass@host:5432/dbname
    db_enabled: bool           # False → use file-based fallback
    lead_score_threshold: int  # default 50
    lead_score_every_n: int    # score lead every N bot messages (default 3)
```

---

### Component 7 — Environment Variables (`.env`)

#### [MODIFY] `.env`

Add these new variables:

```env
# ─── PostgreSQL / Supabase ──────────────────────────────────────────────────────
# For Supabase: copy from Settings → Database → Connection String → URI mode
DATABASE_URL=postgresql://postgres.[project-ref]:[password]@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
# To switch to self-hosted: DATABASE_URL=postgresql://user:pass@localhost:5432/wmcnc

# ─── Lead Scoring ───────────────────────────────────────────────────────────────
LEAD_SCORE_THRESHOLD=50
LEAD_SCORE_EVERY_N_MESSAGES=3
DB_ENABLED=true
```

---

### Component 8 — Next.js Dashboard API Integration

#### [MODIFY] `frontenddashboard/leadsdashboard/src/context/DashboardContext.tsx`

Replace static `coreLeads` and `initialCriteria` arrays with API calls to the FastAPI backend:

```typescript
// Replace static data with:
useEffect(() => {
  fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/leads`)
    .then(r => r.json())
    .then(data => setLeads(data.leads));
}, []);

useEffect(() => {
  fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/scoring-rules`)
    .then(r => r.json())
    .then(data => setCriteriaList(data.rules));
}, []);
```

#### [NEW] `frontenddashboard/leadsdashboard/.env.local`
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Component 9 — Database Migration Script

#### [NEW] `scripts/run_migrations.py`

One-time setup script:

```python
# python scripts/run_migrations.py
# Reads DATABASE_URL, runs 001_initial_schema.sql
# Idempotent — safe to re-run
```

#### [NEW] `scripts/seed_rules.py`

Seeds the default 8 lead scoring rules from `DashboardContext.tsx` into Supabase.

---

## Dependencies to Add

```
# requirements.txt additions:
psycopg2-binary    # PostgreSQL driver (sync)  ← swap to asyncpg for full async later
python-jose        # optional: for dashboard JWT auth
```

---

## Verification Plan

### Automated Tests
```bash
# 1. Unit: DB layer with a test Supabase project
python -m pytest tests/test_db_users.py
python -m pytest tests/test_db_messages.py
python -m pytest tests/test_lead_scorer.py

# 2. Integration: simulate WhatsApp webhook
curl -X POST http://localhost:8000/whatsapp/webhook \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/wa_message.json
```

### Manual Verification
1. Send "hi" to WhatsApp → row appears in `users` and `conversations` tables in Supabase dashboard
2. Complete language selection → `users.language` and `users.conversation_state` updated
3. Send name+address → `users.name`, `users.city` populated
4. Send 3 FAQ questions → 3 rows in `messages`, lead score updated in `lead_analysis`
5. Open Next.js dashboard → Leads page shows real data from API, not mock data
6. Create/edit a scoring rule → rule persists after page refresh (confirmed via Supabase table)

### Self-Host Migration Verification
1. Spin up PostgreSQL locally: `docker run -e POSTGRES_PASSWORD=pass -p 5432:5432 postgres:16`
2. Change `DATABASE_URL` in `.env`
3. Run `python scripts/run_migrations.py`
4. Start server — all functionality works identically

---

## Open Questions

> [!IMPORTANT]
> **Q1: Authentication for Dashboard API?**
> Currently the dashboard uses no auth. Should the `/api/*` routes require a secret token or Basic Auth header to prevent public access? A simple `DASHBOARD_API_KEY` env var bearer token is recommended. 
>yes Add basic auth header

> [!IMPORTANT]
> **Q2: Lead Scoring  Trigger Strategy**
> Option A: Score after every 20 minutes (background task)


> **Q3: Web widget sessions**
> Web sessions use `session_id` like `"abc123"` (no phone number). Should we store these in `users.phone_number` as-is (synthetic ID), or create a separate `web_sessions` table?
> yes in users.phone_number as-is (synthetic ID)

> [!NOTE]
> **Q4: Existing file-based sessions**
> There are existing `artifacts/flow_state/*.json` files. Should we **migrate** them to PostgreSQL on startup, or simply **ignore** them and let old sessions reset?
>let old sessions reset?

> [!WARNING]
> **Q5: Supabase connection pooling**
> Supabase in "Transaction mode" (port 6543) doesn't support server-side prepared statements. The DB client must set `prepare_threshold=0` on the psycopg2 pool. This is already accounted for in the plan.

---

## Implementation Order (Phases)

| Phase | What | Est. Effort |
|-------|------|------------|
| **P1** | DB layer (`src/db/`) + migration SQL | ~1 day |
| **P2** | Wire `conversation_flow.py` to DB | ~0.5 day |
| **P3** | Wire `rag_pipeline.py` (bot message logging) | ~0.5 day |
| **P4** | Lead scoring engine (`src/lead/scorer.py`) | ~1 day |
| **P5** | Dashboard API routes (`src/api/`) | ~1 day |
| **P6** | Next.js dashboard: replace mock data with API calls | ~1 day |
| **P7** | Testing + migration scripts | ~0.5 day |
| **Total** | | **~5.5 days** |
