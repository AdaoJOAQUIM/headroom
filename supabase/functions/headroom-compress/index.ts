// Headroom — headroom-compress Edge Function
//
// The Headroom compression engine, running inside Supabase. One endpoint, two
// actions:
//
//   POST { "action": "compress", "content": "...", "store": false,
//          "user_id"?: "...", "project"?: "..." }
//     -> { compressed, mode, ratio, original_chars, compressed_chars,
//          refs?  (when store=false),
//          ref_id? (when store=true — refs persisted to headroom_compression_store) }
//
//   POST { "action": "restore", "compressed": "...", "mode": "...",
//          "refs": [...] }                       // stateless restore
//   POST { "action": "restore", "compressed": "...", "ref_id": "..." }  // CCR restore
//     -> { original }
//
// Reversible structure-preserving compression (JSON + log/text). The heavy ML
// path (Magika detection, Kompress neural compressor) is NOT here — it cannot
// run in a Deno Edge Function and stays local in Headroom. See ../README.md.
//
// Deploy:  supabase functions deploy headroom-compress
// Schema:  sql/create_compression_store_supabase.sql (only needed for store=true)

import { compress, restore } from "../_shared/headroom_compress.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_KEY =
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? Deno.env.get("SUPABASE_ANON_KEY") ?? "";
const STORE_TABLE = "headroom_compression_store";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

async function persistRefs(row: Record<string, unknown>): Promise<string | null> {
  const resp = await fetch(`${SUPABASE_URL}/rest/v1/${STORE_TABLE}`, {
    method: "POST",
    headers: {
      apikey: SERVICE_KEY,
      Authorization: `Bearer ${SERVICE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(row),
  });
  if (!resp.ok) return null;
  const rows = await resp.json();
  return Array.isArray(rows) && rows[0]?.ref_id ? String(rows[0].ref_id) : null;
}

async function fetchRefs(refId: string): Promise<{ mode: string; refs: unknown[] } | null> {
  const resp = await fetch(
    `${SUPABASE_URL}/rest/v1/${STORE_TABLE}?ref_id=eq.${encodeURIComponent(refId)}&limit=1`,
    {
      headers: { apikey: SERVICE_KEY, Authorization: `Bearer ${SERVICE_KEY}` },
    },
  );
  if (!resp.ok) return null;
  const rows = await resp.json();
  if (!Array.isArray(rows) || rows.length === 0) return null;
  return { mode: rows[0].mode, refs: rows[0].refs ?? [] };
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return json({ error: "method not allowed" }, 405);

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const action = body.action;

  // ---- compress ----
  if (action === "compress") {
    if (typeof body.content !== "string") {
      return json({ error: "content (string) is required" }, 400);
    }
    const result = compress(body.content, (body.options as object) ?? {});
    const payload: Record<string, unknown> = {
      compressed: result.compressed,
      mode: result.mode,
      ratio: result.ratio,
      original_chars: result.originalChars,
      compressed_chars: result.compressedChars,
    };

    if (body.store === true) {
      const refId = await persistRefs({
        mode: result.mode,
        refs: result.refs,
        user_id: body.user_id ?? null,
        project: body.project ?? null,
        original_chars: result.originalChars,
        compressed_chars: result.compressedChars,
        ratio: result.ratio,
      });
      if (!refId) {
        return json({ error: "failed to persist refs (is the schema applied?)" }, 502);
      }
      payload.ref_id = refId;
    } else {
      payload.refs = result.refs; // stateless: caller keeps the reference table
    }
    return json(payload);
  }

  // ---- restore ----
  if (action === "restore") {
    if (typeof body.compressed !== "string") {
      return json({ error: "compressed (string) is required" }, 400);
    }
    let mode = body.mode as string | undefined;
    let refs = body.refs as unknown[] | undefined;

    if (typeof body.ref_id === "string") {
      const stored = await fetchRefs(body.ref_id);
      if (!stored) return json({ error: "ref_id not found" }, 404);
      mode = stored.mode;
      refs = stored.refs;
    }
    if (!mode) return json({ error: "mode or ref_id is required for restore" }, 400);

    try {
      const original = restore(body.compressed, mode, refs ?? []);
      return json({ original });
    } catch (e) {
      return json({ error: `restore failed: ${(e as Error).message}` }, 400);
    }
  }

  return json({ error: "action must be 'compress' or 'restore'" }, 400);
});
