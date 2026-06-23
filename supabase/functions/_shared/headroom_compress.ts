// Headroom compression engine — TypeScript port (runs inside Supabase Edge Functions).
//
// This is a faithful, dependency-free port of the *reversible* subset of
// Headroom's compression: structure-preserving JSON compression + consecutive
// line de-duplication for logs/text. It mirrors the strategy of
// headroom/compression/handlers/json_handler.py — keep the navigational
// structure (keys, brackets, booleans, nulls, identifiers, short values) and
// elide the bulky values into a reference table so the original is exactly
// reconstructable (Headroom's CCR — Compression Context Retrieval — model).
//
// What is intentionally NOT ported: the ML pieces (Magika content detection,
// the Kompress neural compressor). Those need a Rust runtime / model weights
// that Supabase Edge Functions cannot host; they stay local in Headroom. See
// supabase/functions/README.md.

export interface CompressResult {
  compressed: string; // compressed payload (string; JSON-encoded for json mode)
  mode: "json" | "text" | "noop";
  refs: unknown[]; // reference table: elided originals, indexed by marker
  originalChars: number;
  compressedChars: number;
  ratio: number; // originalChars / compressedChars (>= 1 means it shrank)
}

export interface CompressOptions {
  // Strings at least this long (and not identifier-like) are elided. Mirrors
  // json_handler.short_value_threshold semantics (keep short values inline).
  minValueLength?: number;
  // Keep the first N items of a long array in full; elide the tail. Mirrors
  // json_handler.max_array_items_full.
  maxArrayItemsFull?: number;
  // Minimum content length before we bother compressing at all.
  minContentLength?: number;
}

const DEFAULTS: Required<CompressOptions> = {
  minValueLength: 40,
  maxArrayItemsFull: 3,
  minContentLength: 100,
};

// Sentinel markers. Chosen to be vanishingly unlikely in real content and to
// survive JSON round-tripping. § is the section sign (§).
const S_MARK = (i: number) => `§S${i}§`; // elided string value
const A_MARK = (i: number) => `§A${i}§`; // elided array tail
const R_MARK_RE = /^§R(\d+)§$/; // line run-length marker

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const HASH_RE = /^[0-9a-f]{32,}$/i; // md5/sha-like hex identifiers

function isIdentifierLike(s: string): boolean {
  // Identifiers carry signal for the LLM and are short — preserve them inline.
  return UUID_RE.test(s) || HASH_RE.test(s) || (!s.includes(" ") && s.length <= 24);
}

function isJsonContent(content: string): boolean {
  const t = content.trimStart();
  return t.startsWith("{") || t.startsWith("[");
}

// --------------------------------------------------------------------------- //
// JSON structural compression (reversible)
// --------------------------------------------------------------------------- //

function transform(value: unknown, refs: unknown[], opts: Required<CompressOptions>): unknown {
  if (typeof value === "string") {
    if (value.length >= opts.minValueLength && !isIdentifierLike(value)) {
      const idx = refs.length;
      refs.push({ t: "s", v: value });
      return S_MARK(idx);
    }
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length > opts.maxArrayItemsFull) {
      const head = value
        .slice(0, opts.maxArrayItemsFull)
        .map((v) => transform(v, refs, opts));
      const idx = refs.length;
      refs.push({ t: "a", v: value.slice(opts.maxArrayItemsFull) }); // original tail
      head.push(A_MARK(idx));
      return head;
    }
    return value.map((v) => transform(v, refs, opts));
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = transform(v, refs, opts); // keys are structure: always preserved
    }
    return out;
  }
  return value; // numbers, booleans, null — small + semantically important
}

function restoreValue(value: unknown, refs: unknown[]): unknown {
  if (typeof value === "string") {
    const m = value.match(/^§S(\d+)§$/);
    if (m) return (refs[Number(m[1])] as { v: unknown }).v;
    return value;
  }
  if (Array.isArray(value)) {
    const last = value[value.length - 1];
    if (typeof last === "string") {
      const am = last.match(/^§A(\d+)§$/);
      if (am) {
        const head = value.slice(0, -1).map((v) => restoreValue(v, refs));
        const tail = (refs[Number(am[1])] as { v: unknown[] }).v.map((v) =>
          restoreValue(v, refs),
        );
        return [...head, ...tail];
      }
    }
    return value.map((v) => restoreValue(v, refs));
  }
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = restoreValue(v, refs);
    }
    return out;
  }
  return value;
}

// --------------------------------------------------------------------------- //
// Text / log compression: collapse runs of identical consecutive lines.
// Fully reversible: "line\n§R<k>§" expands back to k copies.
// --------------------------------------------------------------------------- //

function compressText(content: string): { compressed: string; refs: unknown[] } {
  const lines = content.split("\n");
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    let run = 1;
    while (i + run < lines.length && lines[i + run] === lines[i]) run++;
    out.push(lines[i]);
    if (run > 1) out.push(`§R${run}§`);
    i += run;
  }
  return { compressed: out.join("\n"), refs: [] };
}

function restoreText(compressed: string): string {
  const lines = compressed.split("\n");
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(R_MARK_RE);
    if (m && out.length > 0) {
      const k = Number(m[1]);
      const prev = out[out.length - 1];
      for (let j = 1; j < k; j++) out.push(prev); // already pushed once
    } else {
      out.push(lines[i]);
    }
  }
  return out.join("\n");
}

// --------------------------------------------------------------------------- //
// Public API
// --------------------------------------------------------------------------- //

export function compress(content: string, options: CompressOptions = {}): CompressResult {
  const opts = { ...DEFAULTS, ...options };
  const originalChars = content.length;

  if (originalChars < opts.minContentLength) {
    return {
      compressed: content,
      mode: "noop",
      refs: [],
      originalChars,
      compressedChars: originalChars,
      ratio: 1,
    };
  }

  if (isJsonContent(content)) {
    try {
      const parsed = JSON.parse(content);
      const refs: unknown[] = [];
      const transformed = transform(parsed, refs, opts);
      const compressed = JSON.stringify(transformed); // compact (no whitespace)
      return {
        compressed,
        mode: "json",
        refs,
        originalChars,
        compressedChars: compressed.length,
        ratio: compressed.length ? originalChars / compressed.length : 1,
      };
    } catch {
      // Not valid JSON after all — fall through to text mode.
    }
  }

  const { compressed } = compressText(content);
  return {
    compressed,
    mode: "text",
    refs: [],
    originalChars,
    compressedChars: compressed.length,
    ratio: compressed.length ? originalChars / compressed.length : 1,
  };
}

export function restore(compressed: string, mode: string, refs: unknown[]): string {
  if (mode === "noop") return compressed;
  if (mode === "text") return restoreText(compressed);
  if (mode === "json") {
    const parsed = JSON.parse(compressed);
    return JSON.stringify(restoreValue(parsed, refs));
  }
  throw new Error(`unknown compression mode: ${mode}`);
}
