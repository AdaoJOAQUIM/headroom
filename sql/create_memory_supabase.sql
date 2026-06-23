-- Supabase SQL: Headroom Cognitive Memory layer
-- Run this in the Supabase SQL Editor (https://supabase.com/dashboard -> SQL Editor).
--
-- This schema turns Supabase from a telemetry sink (see create_proxy_telemetry_v2.sql)
-- into the persistence + retrieval substrate for Headroom's memory system. It backs
-- three roles at once, all inside Supabase:
--
--   * MemoryStore   -> table `headroom_memories`   (persistent, temporally-aware memory)
--   * VectorIndex   -> column `embedding vector`   + RPC `match_headroom_memories`
--                      ("reconstruction du contexte utile" / retrieval)
--   * GraphStore    -> tables `headroom_entities` + `headroom_relationships`
--                      (knowledge graph / patterns)
--
-- Heavy compression stays local in Headroom (Rust/Python). Supabase stores the
-- compressed/extracted result and serves light retrieval — the "hybrid" shape.
--
-- The columns mirror headroom/memory/models.py::Memory.to_dict() one-to-one so the
-- Python SupabaseMemoryStore adapter can round-trip without a translation layer.

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector: similarity search
CREATE EXTENSION IF NOT EXISTS pgcrypto;     -- gen_random_uuid()

-- Default embedding dimension matches MemoryConfig.vector_dimension (384,
-- e.g. all-MiniLM-L6-v2 / the bundled ONNX embedder). If you switch to a
-- different embedder, recreate the column and RPC with the matching size.

-- ---------------------------------------------------------------------------
-- Memory store
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS headroom_memories (
    -- Identity
    id            text PRIMARY KEY,
    content       text NOT NULL DEFAULT '',

    -- Hierarchical scoping (user_id required; narrower scopes optional)
    user_id       text NOT NULL DEFAULT '',
    session_id    text,
    agent_id      text,
    turn_id       text,

    -- Temporal awareness (bitemporal)
    created_at    timestamptz NOT NULL DEFAULT now(),
    valid_from    timestamptz NOT NULL DEFAULT now(),
    valid_until   timestamptz,                 -- NULL = current/active

    -- Classification
    importance    real NOT NULL DEFAULT 0.5,   -- 0.0 - 1.0

    -- Lineage (supersession + bubbling)
    supersedes      text,
    superseded_by   text,
    promoted_from   text,
    promotion_chain jsonb NOT NULL DEFAULT '[]'::jsonb,

    -- Access tracking
    access_count  integer NOT NULL DEFAULT 0,
    last_accessed timestamptz,

    -- Entity references (knowledge-graph links)
    entity_refs   jsonb NOT NULL DEFAULT '[]'::jsonb,

    -- Vector embedding for similarity search (pgvector)
    embedding     vector(384),

    -- Free-form metadata (project id, tags, source, reuse counters, ...)
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Scope / temporal / importance access paths used by MemoryFilter -> PostgREST.
CREATE INDEX IF NOT EXISTS idx_hm_user        ON headroom_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_hm_session     ON headroom_memories(session_id);
CREATE INDEX IF NOT EXISTS idx_hm_agent       ON headroom_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_hm_created     ON headroom_memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hm_current     ON headroom_memories(user_id) WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS idx_hm_importance  ON headroom_memories(importance);

-- Approximate-nearest-neighbour index for cosine similarity. ivfflat needs data
-- before it is effective; for small/medium sets PostgreSQL will seq-scan, which
-- is fine. Recreate with a tuned `lists` value once the table is populated.
CREATE INDEX IF NOT EXISTS idx_hm_embedding
    ON headroom_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- Knowledge graph (entities + relationships)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS headroom_entities (
    id          text PRIMARY KEY,
    name        text NOT NULL DEFAULT '',
    entity_type text NOT NULL DEFAULT '',
    user_id     text NOT NULL DEFAULT '',
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_he_user ON headroom_entities(user_id);
CREATE INDEX IF NOT EXISTS idx_he_name ON headroom_entities(user_id, name);

CREATE TABLE IF NOT EXISTS headroom_relationships (
    id               text PRIMARY KEY,
    source_entity_id text NOT NULL,
    target_entity_id text NOT NULL,
    relation_type    text NOT NULL DEFAULT '',
    user_id          text NOT NULL DEFAULT '',
    memory_id        text,
    weight           real NOT NULL DEFAULT 1.0,
    metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_hr_source ON headroom_relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_hr_target ON headroom_relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_hr_user   ON headroom_relationships(user_id);

-- ---------------------------------------------------------------------------
-- Retrieval RPC: cosine-similarity search, scope-filtered, current-only.
-- Called by the Python SupabaseVectorIndex and by the memory-recall Edge Function.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_headroom_memories(
    query_embedding vector(384),
    match_count     integer DEFAULT 10,
    min_similarity  real    DEFAULT 0.0,
    p_user_id       text    DEFAULT NULL,
    p_session_id    text    DEFAULT NULL,
    include_superseded boolean DEFAULT false
)
RETURNS TABLE (
    id          text,
    content     text,
    user_id     text,
    session_id  text,
    agent_id    text,
    turn_id     text,
    importance  real,
    entity_refs jsonb,
    metadata    jsonb,
    created_at  timestamptz,
    similarity  real
)
LANGUAGE sql STABLE
AS $$
    SELECT
        m.id, m.content, m.user_id, m.session_id, m.agent_id, m.turn_id,
        m.importance, m.entity_refs, m.metadata, m.created_at,
        (1 - (m.embedding <=> query_embedding))::real AS similarity
    FROM headroom_memories m
    WHERE m.embedding IS NOT NULL
      AND (p_user_id    IS NULL OR m.user_id = p_user_id)
      AND (p_session_id IS NULL OR m.session_id = p_session_id)
      AND (include_superseded OR m.valid_until IS NULL)
      AND (1 - (m.embedding <=> query_embedding)) >= min_similarity
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------
-- The anon role gets full CRUD here because Headroom is a local-first developer
-- tool: the anon key lives on the developer's own machine and the project is the
-- user's own. If you expose this to untrusted clients, tighten these policies
-- (e.g. per-user RLS keyed on auth.uid()) and use the service_role key only from
-- the Edge Functions.
ALTER TABLE headroom_memories      ENABLE ROW LEVEL SECURITY;
ALTER TABLE headroom_entities      ENABLE ROW LEVEL SECURITY;
ALTER TABLE headroom_relationships ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['headroom_memories','headroom_entities','headroom_relationships']
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS anon_all ON %I', t);
        EXECUTE format(
            'CREATE POLICY anon_all ON %I FOR ALL TO anon USING (true) WITH CHECK (true)', t);
        EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO anon', t);
    END LOOP;
END $$;

GRANT EXECUTE ON FUNCTION match_headroom_memories TO anon;
