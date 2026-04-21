// migrate.js — runs on every server startup.
// Creates all tables if they don't exist (safe to re-run).
// Requires DATABASE_URL env var (Supabase: Settings → Database → URI).

const { Pool } = require("pg");

const MIGRATIONS = `
-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- customers
CREATE TABLE IF NOT EXISTS customers (
  id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  name              TEXT,
  phone             TEXT,
  email             TEXT,
  tier              TEXT,
  years_as_customer INTEGER,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- bills (customer billing data queried by AI)
CREATE TABLE IF NOT EXISTS bills (
  id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  customer_id    UUID        REFERENCES customers(id) ON DELETE SET NULL,
  customer_phone TEXT,
  customer_name  TEXT,
  bill_month     TEXT,
  amount         NUMERIC(10,2),
  due_date       DATE,
  paid_date      DATE,
  status         TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- calls
CREATE TABLE IF NOT EXISTS calls (
  id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  customer_phone   TEXT,
  customer_name    TEXT,
  tier             TEXT,
  priority         TEXT        DEFAULT 'low',
  duration_seconds INTEGER,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- messages (live transcript)
CREATE TABLE IF NOT EXISTS messages (
  id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  call_id    UUID        NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  role       TEXT        NOT NULL,
  content    TEXT        NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- analysis (AI results)
CREATE TABLE IF NOT EXISTS analysis (
  id                UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  call_id           UUID        NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
  emotion           TEXT,
  intent            TEXT,
  priority          TEXT,
  suggested_actions TEXT[],
  suggested_reply   TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- knowledge_base (RAG chunks + embeddings)
CREATE TABLE IF NOT EXISTS knowledge_base (
  id         UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  content    TEXT        NOT NULL,
  source     TEXT,
  embedding  VECTOR(768),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vector similarity index (skip if already exists)
CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx
  ON knowledge_base
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- match_knowledge RPC used by ragService.js
CREATE OR REPLACE FUNCTION match_knowledge(
  query_embedding VECTOR(768),
  match_count     INT DEFAULT 3
)
RETURNS TABLE (id UUID, content TEXT, source TEXT, similarity FLOAT)
LANGUAGE sql STABLE AS $$
  SELECT id, content, source,
         1 - (embedding <=> query_embedding) AS similarity
  FROM knowledge_base
  WHERE embedding IS NOT NULL
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Disable RLS (internal tool — prevents empty realtime payloads)
ALTER TABLE customers      DISABLE ROW LEVEL SECURITY;
ALTER TABLE bills          DISABLE ROW LEVEL SECURITY;
ALTER TABLE calls          DISABLE ROW LEVEL SECURITY;
ALTER TABLE messages       DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis       DISABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base DISABLE ROW LEVEL SECURITY;

-- Enable Realtime (errors ignored if already added)
DO $$
BEGIN
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE calls;
  EXCEPTION WHEN others THEN NULL; END;
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE messages;
  EXCEPTION WHEN others THEN NULL; END;
  BEGIN
    ALTER PUBLICATION supabase_realtime ADD TABLE analysis;
  EXCEPTION WHEN others THEN NULL; END;
END $$;
`;

async function runMigrations() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.warn("[migrate] DATABASE_URL not set — skipping auto-migration.");
    console.warn("[migrate] Add it in Render: Supabase → Settings → Database → URI");
    return;
  }

  const pool = new Pool({
    connectionString: url,
    ssl: { rejectUnauthorized: false },
    // Single connection — close immediately after migration
    max: 1,
    idleTimeoutMillis: 5000,
  });

  try {
    console.log("[migrate] Running database migrations...");
    await pool.query(MIGRATIONS);
    console.log("[migrate] All tables ready ✓");
  } catch (err) {
    // Log but don't crash the server — tables may already exist
    console.error("[migrate] Migration error:", err.message);
  } finally {
    await pool.end();
  }
}

module.exports = { runMigrations };
