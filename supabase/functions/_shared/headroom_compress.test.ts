// Reversibility + ratio tests for the Headroom compression engine.
//
// Run with Deno:  deno test supabase/functions/_shared/headroom_compress.test.ts
// Run with Node:  node --experimental-strip-types supabase/functions/_shared/headroom_compress.test.ts
//
// The contract under test: restore(compress(x)) reconstructs x exactly (JSON is
// compared semantically; text byte-for-byte), and real payloads shrink.

import { compress, restore } from "./headroom_compress.ts";

let failures = 0;

function roundtrips(name: string, original: string, expectShrink = false): void {
  const r = compress(original);
  const back = restore(r.compressed, r.mode, r.refs);
  const ok =
    r.mode === "json"
      ? JSON.stringify(JSON.parse(back)) === JSON.stringify(JSON.parse(original))
      : back === original;
  const shrinkOk = !expectShrink || r.ratio > 1.2;
  if (ok && shrinkOk) {
    console.log(`ok   ${name}  ratio=${r.ratio.toFixed(2)} mode=${r.mode}`);
  } else {
    failures++;
    console.error(`FAIL ${name} (ok=${ok}, ratio=${r.ratio.toFixed(2)})`);
  }
}

const big = "x".repeat(60);

roundtrips(
  "json elides long values, preserves structure + identifiers",
  JSON.stringify({
    name: "Alice",
    id: "usr_123",
    uuid: "550e8400-e29b-41d4-a716-446655440000",
    bio: "A long biography that exceeds the value threshold and must be reversible.",
    active: true,
    score: 42,
    nested: { note: big },
  }),
  true,
);

roundtrips(
  "json elides long array tail",
  JSON.stringify({
    items: Array.from({ length: 20 }, (_, i) => ({ i, label: `item-${i}-${big}` })),
    count: 20,
  }),
  true,
);

roundtrips(
  "top-level array of objects",
  JSON.stringify(Array.from({ length: 8 }, (_, i) => ({ k: i, v: `value ${i} ${big}` }))),
  true,
);

roundtrips("short json is a noop", JSON.stringify({ a: 1, b: true }));

roundtrips(
  "text log run-length is reversible",
  Array(30).fill("ERROR: connection refused").join("\n") +
    "\nunique\n" +
    Array(10).fill("retrying").join("\n"),
  true,
);

roundtrips("short text is a noop", "hello world");

if (failures > 0) {
  console.error(`\n${failures} test(s) failed`);
  // Deno: throw to fail; Node: set exit code.
  if (typeof (globalThis as { process?: { exit?: (n: number) => void } }).process !== "undefined") {
    (globalThis as { process: { exitCode: number } }).process.exitCode = 1;
  } else {
    throw new Error(`${failures} test(s) failed`);
  }
} else {
  console.log("\nall reversibility tests passed");
}
