CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(100),
    email VARCHAR(255),
    language VARCHAR(20),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100),
    address TEXT,
    first_seen TIMESTAMP NOT NULL DEFAULT now(),
    last_seen TIMESTAMP NOT NULL DEFAULT now(),
    status VARCHAR(20) NOT NULL DEFAULT 'New',
    conversation_state VARCHAR(30) NOT NULL DEFAULT 'LANGUAGE_SELECTION',
    current_conversation_id UUID,
    source VARCHAR(30) NOT NULL DEFAULT 'WEB',
    assigned_to VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone_number);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_last_seen ON users(last_seen DESC);

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMP NOT NULL DEFAULT now(),
    ended_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    source VARCHAR(30) NOT NULL DEFAULT 'WEB',
    lead_score INT NOT NULL DEFAULT 0,
    summary_generated BOOLEAN NOT NULL DEFAULT false,
    contact_shared BOOLEAN NOT NULL DEFAULT false,
    closed_reason VARCHAR(200),
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON conversations(started_at DESC);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_users_current_conversation'
    ) THEN
        ALTER TABLE users
            ADD CONSTRAINT fk_users_current_conversation
            FOREIGN KEY (current_conversation_id)
            REFERENCES conversations(id)
            ON DELETE SET NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender VARCHAR(10) NOT NULL,
    message_type VARCHAR(20) NOT NULL DEFAULT 'text',
    message TEXT NOT NULL,
    language VARCHAR(20),
    timestamp TIMESTAMP NOT NULL DEFAULT now(),
    tokens INT,
    ai_model VARCHAR(100),
    latency_ms INT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);

CREATE TABLE IF NOT EXISTS lead_score_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    description TEXT,
    weight INT NOT NULL DEFAULT 10,
    enabled BOOLEAN NOT NULL DEFAULT true,
    priority INT NOT NULL DEFAULT 0,
    matching_type VARCHAR(50) NOT NULL DEFAULT 'AI Detected (Recommended)',
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lead_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    intent VARCHAR(50),
    urgency VARCHAR(20),
    interest_level VARCHAR(20),
    product_interest VARCHAR(200),
    budget VARCHAR(100),
    timeline VARCHAR(100),
    sentiment VARCHAR(20),
    language VARCHAR(20),
    lead_score INT NOT NULL DEFAULT 0,
    qualified BOOLEAN NOT NULL DEFAULT false,
    confidence DECIMAL(5,2),
    recommended_action TEXT,
    matched_rule_ids UUID[],
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lead_analysis_conversation_id ON lead_analysis(conversation_id);
CREATE INDEX IF NOT EXISTS idx_lead_analysis_lead_score ON lead_analysis(lead_score DESC);
CREATE INDEX IF NOT EXISTS idx_lead_analysis_qualified ON lead_analysis(qualified);

CREATE TABLE IF NOT EXISTS conversation_summary (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    key_points JSONB,
    customer_requirements JSONB,
    follow_up_needed BOOLEAN NOT NULL DEFAULT false,
    generated_at TIMESTAMP NOT NULL DEFAULT now(),
    updated_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_conversation_id ON events(conversation_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

INSERT INTO lead_score_rules (rule_name, description, weight, priority, matching_type)
SELECT * FROM (
    VALUES
        ('Asked About Price', 'Lead asked about price or quotation', 20, 1, 'AI Detected (Recommended)'),
        ('Product Interest', 'Lead showed interest in specific machine model', 15, 2, 'AI Detected (Recommended)'),
        ('Asked About Features', 'Lead asked about features or specifications', 10, 3, 'AI Detected (Recommended)'),
        ('Timeline Mentioned', 'Lead mentioned purchase timeline', 15, 4, 'AI Detected (Recommended)'),
        ('Asked for Brochure / Catalog', 'Lead requested brochure or catalog', 10, 5, 'AI Detected (Recommended)'),
        ('Provided Location', 'Lead shared location or city', 5, 6, 'AI Detected (Recommended)'),
        ('Multiple Interactions', 'Lead came back for another conversation', 10, 7, 'AI Detected (Recommended)'),
        ('Qualified by AI', 'AI marked lead as highly qualified', 25, 8, 'AI Detected (Recommended)')
) AS seed(rule_name, description, weight, priority, matching_type)
WHERE NOT EXISTS (
    SELECT 1 FROM lead_score_rules existing WHERE existing.rule_name = seed.rule_name
);
