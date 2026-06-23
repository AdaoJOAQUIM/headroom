# Headroom in Supabase — module

This directory is the **Supabase-native module of Headroom**: as much of
Headroom as the Supabase platform can actually run, packaged to deploy as a
unit. It complements the Python `SupabaseMemoryStore` / `SupabaseVectorIndex`
backends (`headroom/memory/adapters/supabase_store.py`).

## What runs *inside* Supabase

| Capability | Where | Reversible |
|------------|-------|------------|
| Cognitive memory (store) | `headroom_memories` table | — |
| Vector retrieval | `match_headroom_memories` RPC + `memory-recall/` | — |
| Knowledge graph | `headroom_entities` / `headroom_relationships` | — |
| **Compression engine** | `headroom-compress/` (+ `_shared/headroom_compress.ts`) | ✅ |
| CCR reference store | `headroom_compression_store` table | — |
| Telemetry | `proxy_telemetry_v2` table | — |

The compression engine is a faithful TypeScript port of the **reversible** part
of Headroom (`headroom/compression/handlers/json_handler.py`): it preserves the
navigational structure (keys, brackets, booleans, nulls, identifiers, short
values) and elides bulky string values / long array tails into a reference
table, so the original is reconstructed exactly. Verified round-trip on JSON and
log/text inputs (2×–18× shrink on typical payloads).

## What cannot run in Supabase (stays local in Headroom)

Supabase Edge Functions are **Deno/TypeScript** with hard CPU/memory/time
limits, and Postgres cannot host arbitrary native code. So the heavy pieces of
Headroom stay on the Headroom side and are **not** ported here:

- the **Rust** compression core and the **Kompress** neural compressor (model weights);
- **Magika** ML content detection;
- the proxy, the MCP server, and the rest of the Python package.

This is the deliberate **hybrid** split: heavy compute local, storage + light
reversible compression + retrieval in Supabase, behind one API.

## Endpoints

### `headroom-compress`
```bash
# compress (stateless — caller keeps refs)
curl -sX POST "$URL/functions/v1/headroom-compress" -H "apikey: $KEY" \
  -H 'content-type: application/json' \
  -d '{"action":"compress","content":"{...large json...}"}'

# compress + persist refs (CCR); returns ref_id instead of refs
curl -sX POST "$URL/functions/v1/headroom-compress" -H "apikey: $KEY" \
  -H 'content-type: application/json' \
  -d '{"action":"compress","content":"...","store":true,"project":"myapp"}'

# restore (stateless)
curl -sX POST "$URL/functions/v1/headroom-compress" -H "apikey: $KEY" \
  -H 'content-type: application/json' \
  -d '{"action":"restore","compressed":"...","mode":"json","refs":[...]}'

# restore via CCR
curl -sX POST "$URL/functions/v1/headroom-compress" -H "apikey: $KEY" \
  -H 'content-type: application/json' \
  -d '{"action":"restore","compressed":"...","ref_id":"<id>"}'
```

### `memory-recall`
Scope-filtered cosine search over `headroom_memories` (see `memory-recall/index.ts`).

## Schema

Apply, in order, in the Supabase SQL Editor (or via `supabase db push`):

1. `sql/create_proxy_telemetry_v2.sql` (already in use)
2. `sql/create_memory_supabase.sql`
3. `sql/create_compression_store_supabase.sql`
