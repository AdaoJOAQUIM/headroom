// Headroom — memory-recall Edge Function
//
// The "hybrid" retrieval endpoint: a single API every project can call to pull
// back the most relevant accumulated memory for a task. Heavy work (embedding,
// compression) stays local in Headroom; this function only does the light part
// — scope-filtered cosine search over headroom_memories via the
// match_headroom_memories RPC — close to the data.
//
// Deploy:
//   supabase functions deploy memory-recall
// Requires the schema from sql/create_memory_supabase.sql and the standard
// SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY function secrets (injected by default).
//
// Request body (POST, application/json):
//   {
//     "query_embedding": number[],   // required: precomputed query vector (dim 384)
//     "match_count":     number,     // optional, default 10
//     "min_similarity":  number,     // optional, default 0.0
//     "user_id":         string,     // optional scope filter
//     "session_id":      string,     // optional scope filter
//     "include_superseded": boolean  // optional, default false
//   }
//
// Response: { "matches": [ { id, content, importance, similarity, ... } ] }

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_KEY =
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? Deno.env.get("SUPABASE_ANON_KEY")!;

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

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return json({ error: "method not allowed" }, 405);

  let payload: Record<string, unknown>;
  try {
    payload = await req.json();
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const embedding = payload.query_embedding;
  if (!Array.isArray(embedding) || embedding.length === 0) {
    return json({ error: "query_embedding (number[]) is required" }, 400);
  }

  const rpcBody = {
    query_embedding: `[${(embedding as number[]).join(",")}]`,
    match_count: typeof payload.match_count === "number" ? payload.match_count : 10,
    min_similarity:
      typeof payload.min_similarity === "number" ? payload.min_similarity : 0.0,
    p_user_id: payload.user_id ?? null,
    p_session_id: payload.session_id ?? null,
    include_superseded: payload.include_superseded === true,
  };

  const resp = await fetch(`${SUPABASE_URL}/rest/v1/rpc/match_headroom_memories`, {
    method: "POST",
    headers: {
      apikey: SERVICE_KEY,
      Authorization: `Bearer ${SERVICE_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(rpcBody),
  });

  if (!resp.ok) {
    return json({ error: "rpc failed", detail: await resp.text() }, 502);
  }

  return json({ matches: await resp.json() });
});
