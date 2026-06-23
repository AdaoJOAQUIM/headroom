-- Supabase SQL: Headroom CCR (Compression Context Retrieval) store
-- Run this in the Supabase SQL Editor after create_memory_supabase.sql.
--
-- Backs the reversible side of the headroom-compress Edge Function: when a
-- caller compresses with `store=true`, the elided originals (the reference
-- table) are persisted here keyed by ref_id, so a later /restore can rebuild
-- the exact original without the client having to hold the refs. This is the
-- Supabase-native equivalent of Headroom's local CCR cache.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS headroom_compression_store (
    ref_id           text PRIMARY KEY DEFAULT gen_random_uuid()::text,
    created_at       timestamptz NOT NULL DEFAULT now(),

    -- Optional scoping so a project can list/expire its own entries.
    user_id          text,
    project          text,

    mode             text NOT NULL,            -- 'json' | 'text' | 'noop'
    refs             jsonb NOT NULL DEFAULT '[]'::jsonb,  -- elided originals

    -- Bookkeeping for telemetry / savings ledger.
    original_chars   integer,
    compressed_chars integer,
    ratio            real
);

CREATE INDEX IF NOT EXISTS idx_hcs_created ON headroom_compression_store(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hcs_user    ON headroom_compression_store(user_id);
CREATE INDEX IF NOT EXISTS idx_hcs_project ON headroom_compression_store(project);

ALTER TABLE headroom_compression_store ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS anon_all ON headroom_compression_store;
CREATE POLICY anon_all ON headroom_compression_store
    FOR ALL TO anon USING (true) WITH CHECK (true);

GRANT SELECT, INSERT, UPDATE, DELETE ON headroom_compression_store TO anon;
