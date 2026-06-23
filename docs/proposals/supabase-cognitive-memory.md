# Headroom × Supabase — Cognitive Memory layer

Status: **Phase 1 implemented** (memory store + vector retrieval) · Phase 2 planned (memory compiler)

## Why

Today Supabase plays exactly one role for Headroom: an **anonymous telemetry
sink** (`proxy_telemetry_v2`, see `sql/create_proxy_telemetry_v2.sql` and
`headroom/telemetry/beacon.py`). Aggregate stats go in, nothing comes back.

The goal of this work is to promote Supabase from *dashboard* to **cognitive
memory substrate** — so the intelligence Headroom accumulates while compressing
context for one project is not lost, but persists and is reusable across every
future project. The token win stops being only *"spend fewer tokens per call"*
and becomes *"don't lose the intelligence accumulated between projects."*

## The three roles, all in Supabase

Headroom already ships a mature, pluggable memory system
(`headroom/memory/`, with `MemoryStore` / `VectorIndex` / `GraphStore`
Protocols in `ports.py`). Rather than invent a parallel stack, this integration
implements those Protocols against Supabase:

| Role | Local default | Supabase backend |
|------|---------------|------------------|
| **Compressor** (token saving) | Rust/Python pipeline | unchanged — stays local |
| **MemoryStore** (persistence) | SQLite | `headroom_memories` table |
| **VectorIndex** (retrieval / context reconstruction) | sqlite-vec / hnsw | `embedding vector` + `match_headroom_memories` RPC |
| **GraphStore** (knowledge graph / patterns) | SQLite graph | `headroom_entities` + `headroom_relationships` |
| **Telemetry** (token-saving metrics) | — | `proxy_telemetry_v2` (already live) |

## Architecture — Hybrid (chosen)

Heavy compression and embedding stay **local** in Headroom (fast Rust path, no
egress of raw content for the compute step). Supabase is the **storage +
light retrieval** tier. This avoids running the heavy compression model inside
Edge Functions while still giving every project one shared memory via a single
API.

```
   Claude Code / any agent
            │
            ▼
   Headroom compression  ──────────►  (local, Rust/Python)  ← token saving
            │ extract knowledge + embed (local)
            ▼
   Supabase Cognitive Memory
   ┌───────────────────────────────────────────────┐
   │ headroom_memories  (MemoryStore + embedding)   │
   │ match_headroom_memories()  (VectorIndex RPC)   │
   │ headroom_entities / _relationships (GraphStore)│
   │ proxy_telemetry_v2 (existing telemetry)        │
   └───────────────────────────────────────────────┘
            ▲
            │ memory-recall Edge Function (single API for all projects)
            ▼
   reconstructed, relevant context → reused in the next project
```

## What ships in Phase 1

- **`sql/create_memory_supabase.sql`** — pgvector schema, scope/temporal
  indexes, the `match_headroom_memories` cosine RPC, knowledge-graph tables,
  and RLS. Columns mirror `Memory.to_dict()` one-for-one.
- **`headroom/memory/adapters/supabase_store.py`** — `SupabaseMemoryStore`
  (full `MemoryStore` protocol over PostgREST) and `SupabaseVectorIndex`
  (embeddings co-located with their rows; search via the RPC).
- **Wiring** — `StoreBackend.SUPABASE` / `VectorBackend.SUPABASE` in
  `config.py`, constructed by `factory.py` like any other backend.
- **`supabase/functions/memory-recall/index.ts`** — the hybrid retrieval API:
  scope-filtered cosine search every project can call with a precomputed query
  vector.
- **Tests** — `tests/test_memory_supabase_store.py` (serialization, filter
  translation, embedding conversion; network-free).

## Configuration

```bash
# 1. Apply the schema in the Supabase SQL Editor:
#    sql/create_memory_supabase.sql
# 2. Point Headroom at the project:
export HEADROOM_SUPABASE_URL="https://<project>.supabase.co"
export HEADROOM_SUPABASE_KEY="<anon-or-service-key>"
# optional: export HEADROOM_SUPABASE_MEMORY_TABLE="headroom_memories"
```

```python
from headroom.memory.config import MemoryConfig, StoreBackend, VectorBackend
from headroom.memory.factory import create_memory_system

config = MemoryConfig(
    store_backend=StoreBackend.SUPABASE,
    vector_backend=VectorBackend.SUPABASE,
)
store, vector, text, embedder, cache = await create_memory_system(config)
```

The default embedding dimension is 384 (matches `MemoryConfig.vector_dimension`
and the bundled ONNX embedder). Change the `vector(384)` column and the RPC
signature together if you switch embedders.

## Security note

The Phase-1 RLS policies grant the `anon` role full CRUD, which suits a
local-first developer tool where the key lives on the developer's own machine
and the project is their own. For any multi-tenant / untrusted-client exposure,
tighten the policies to per-user RLS (`auth.uid()`) and restrict writes to the
`service_role` used only inside the Edge Functions.

## Phase 2 — the "memory compiler" (planned)

Phase 1 makes Supabase the durable substrate (compressor + store + retrieval).
Phase 2 turns each project into reusable knowledge on top of it:

1. **Extraction** — after each interaction, distil durable facts/patterns
   (`headroom/memory/extraction.py` already does this locally) and persist them
   as `headroom_memories` rows with graph links.
2. **Cross-project recall** — at the start of a new project, `memory-recall`
   reconstructs the relevant accumulated context instead of starting cold.
3. **Reinforcement** — bump `importance` / `access_count` on reuse so the most
   valuable memories surface first (fields already in the schema).

This is the version that matches the "Token Multiplier" intent: not just fewer
tokens per call, but compounding reuse of accumulated intelligence.
