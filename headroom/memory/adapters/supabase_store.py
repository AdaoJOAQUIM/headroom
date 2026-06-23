"""Supabase-backed memory adapters (MemoryStore + VectorIndex).

These adapters persist Headroom's hierarchical memory in Supabase (PostgreSQL +
pgvector) instead of local SQLite, so accumulated intelligence survives across
projects and machines — the "memory in Supabase" / Token Multiplier shape.

Design notes
------------
* Transport is PostgREST (Supabase's auto-generated REST API) over ``httpx``,
  mirroring the telemetry beacon's dependency footprint. No ``supabase-py`` SDK
  is required.
* The table schema is defined in ``sql/create_memory_supabase.sql`` and mirrors
  ``Memory.to_dict()`` column-for-column, so serialization is a thin mapping.
* Heavy compression/extraction stays local in Headroom; Supabase stores the
  result and serves cosine retrieval via the ``match_headroom_memories`` RPC.
* Configuration resolves from ``MemoryConfig`` first, then environment
  (``HEADROOM_SUPABASE_URL`` / ``HEADROOM_SUPABASE_KEY`` /
  ``HEADROOM_SUPABASE_MEMORY_TABLE``).

Only the standard ``httpx`` dependency is needed at runtime; it is imported
lazily so importing this module never hard-fails.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from headroom.memory.models import Memory, ScopeLevel
from headroom.memory.ports import (
    MemoryFilter,
    VectorFilter,
    VectorSearchResult,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a core dep, guard mirrors models.py
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from headroom.memory.config import MemoryConfig

_DEFAULT_TABLE = "headroom_memories"
_MATCH_RPC = "match_headroom_memories"

# Columns PostgREST stores as native JSON (sent/received as Python objects, not
# json.dumps'd strings) versus the SQLite adapter which stores them as TEXT.
_JSON_COLUMNS = frozenset({"promotion_chain", "entity_refs", "metadata"})

_VALID_ORDER_COLUMNS = frozenset({"created_at", "importance", "access_count", "last_accessed"})


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase URL/key cannot be resolved."""


def _resolve_credentials(config: MemoryConfig | None) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, table) from config, falling back to env.

    Raises:
        SupabaseConfigError: if URL or key is missing.
    """
    url = (
        getattr(config, "supabase_url", None) or os.environ.get("HEADROOM_SUPABASE_URL") or ""
    ).rstrip("/")
    key = getattr(config, "supabase_key", None) or os.environ.get("HEADROOM_SUPABASE_KEY") or ""
    table = (
        getattr(config, "supabase_table", None)
        or os.environ.get("HEADROOM_SUPABASE_MEMORY_TABLE")
        or _DEFAULT_TABLE
    )
    if not url or not key:
        raise SupabaseConfigError(
            "Supabase memory backend requires a URL and key. Set them via "
            "MemoryConfig(supabase_url=..., supabase_key=...) or the env vars "
            "HEADROOM_SUPABASE_URL and HEADROOM_SUPABASE_KEY. Apply the schema "
            "first with sql/create_memory_supabase.sql."
        )
    return url, key, table


def _embedding_to_pg(embedding: Any) -> str | None:
    """Serialize a numpy/list embedding to pgvector's bracketed-literal form."""
    if embedding is None:
        return None
    values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _embedding_from_pg(raw: Any) -> Any:
    """Parse a pgvector value (string literal or list) back into a numpy array."""
    if raw is None or np is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    return np.array(raw, dtype=np.float32)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    # PostgREST returns e.g. "2026-06-23T10:06:00+00:00"; fromisoformat handles it.
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class SupabaseMemoryStore:
    """A :class:`~headroom.memory.ports.MemoryStore` backed by Supabase PostgREST."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._base_url, self._key, self._table = _resolve_credentials(config)
        self._rest = f"{self._base_url}/rest/v1"

    # -- HTTP plumbing ------------------------------------------------------

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str]] | None = None,
        json_body: Any = None,
        prefer: str | None = None,
    ) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx ships with the bundle
            raise SupabaseConfigError(
                "httpx is required for the Supabase memory backend "
                "(pip install httpx or the headroom-ai[proxy] bundle)."
            ) from exc

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.request(
                method,
                f"{self._rest}{path}",
                params=params,
                json=json_body,
                headers=self._headers(prefer=prefer),
            )
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        ctype = resp.headers.get("content-type", "")
        return resp.json() if "application/json" in ctype else resp.text

    # -- Serialization ------------------------------------------------------

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        """Convert a Memory to a PostgREST row (JSON-native columns stay objects)."""
        return {
            "id": memory.id,
            "content": memory.content,
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "agent_id": memory.agent_id,
            "turn_id": memory.turn_id,
            "created_at": _iso(memory.created_at),
            "valid_from": _iso(memory.valid_from),
            "valid_until": _iso(memory.valid_until),
            "importance": memory.importance,
            "supersedes": memory.supersedes,
            "superseded_by": memory.superseded_by,
            "promoted_from": memory.promoted_from,
            "promotion_chain": list(memory.promotion_chain),
            "access_count": memory.access_count,
            "last_accessed": _iso(memory.last_accessed),
            "entity_refs": list(memory.entity_refs),
            "embedding": _embedding_to_pg(memory.embedding),
            "metadata": dict(memory.metadata),
        }

    def _row_to_memory(self, row: dict[str, Any]) -> Memory:
        """Convert a PostgREST row back into a Memory."""

        def _json(value: Any, default: Any) -> Any:
            # jsonb columns arrive parsed; tolerate stringified values defensively.
            if value is None:
                return default
            if isinstance(value, str):
                return json.loads(value)
            return value

        return Memory(
            id=row["id"],
            content=row.get("content", ""),
            user_id=row.get("user_id", ""),
            session_id=row.get("session_id"),
            agent_id=row.get("agent_id"),
            turn_id=row.get("turn_id"),
            created_at=_parse_dt(row.get("created_at")) or datetime.utcnow(),
            valid_from=_parse_dt(row.get("valid_from")) or datetime.utcnow(),
            valid_until=_parse_dt(row.get("valid_until")),
            importance=row.get("importance", 0.5),
            supersedes=row.get("supersedes"),
            superseded_by=row.get("superseded_by"),
            promoted_from=row.get("promoted_from"),
            promotion_chain=_json(row.get("promotion_chain"), []),
            access_count=row.get("access_count", 0),
            last_accessed=_parse_dt(row.get("last_accessed")),
            entity_refs=_json(row.get("entity_refs"), []),
            embedding=_embedding_from_pg(row.get("embedding")),
            metadata=_json(row.get("metadata"), {}),
        )

    # -- Filter translation -------------------------------------------------

    def _build_query_params(self, filter: MemoryFilter) -> list[tuple[str, str]]:
        """Translate a MemoryFilter into PostgREST query parameters.

        Mirrors SQLiteMemoryStore._build_query_conditions so both backends agree
        on scope/temporal/importance/lineage semantics.
        """
        params: list[tuple[str, str]] = []

        # Hierarchical scope (same precedence as the SQLite adapter).
        if filter.user_id is not None:
            params.append(("user_id", f"eq.{filter.user_id}"))
            if filter.session_id is not None:
                params.append(("session_id", f"eq.{filter.session_id}"))
                if filter.agent_id is not None:
                    params.append(("agent_id", f"eq.{filter.agent_id}"))
                    if filter.turn_id is not None:
                        params.append(("turn_id", f"eq.{filter.turn_id}"))
            elif filter.agent_id is not None:
                params.append(("agent_id", f"eq.{filter.agent_id}"))
            elif filter.turn_id is not None:
                params.append(("turn_id", f"eq.{filter.turn_id}"))
        elif filter.session_id is not None:
            params.append(("session_id", f"eq.{filter.session_id}"))
        elif filter.agent_id is not None:
            params.append(("agent_id", f"eq.{filter.agent_id}"))
        elif filter.turn_id is not None:
            params.append(("turn_id", f"eq.{filter.turn_id}"))

        # Explicit scope-level filtering via a PostgREST `or=(...)` group.
        if filter.scope_levels:
            groups = []
            for level in filter.scope_levels:
                if level == ScopeLevel.USER:
                    groups.append("and(session_id.is.null,agent_id.is.null,turn_id.is.null)")
                elif level == ScopeLevel.SESSION:
                    groups.append("and(session_id.not.is.null,agent_id.is.null,turn_id.is.null)")
                elif level == ScopeLevel.AGENT:
                    groups.append("and(agent_id.not.is.null,turn_id.is.null)")
                elif level == ScopeLevel.TURN:
                    groups.append("turn_id.not.is.null")
            if groups:
                params.append(("or", "(" + ",".join(groups) + ")"))

        # Temporal filters.
        if filter.created_after is not None:
            params.append(("created_at", f"gte.{filter.created_after.isoformat()}"))
        if filter.created_before is not None:
            params.append(("created_at", f"lte.{filter.created_before.isoformat()}"))
        if filter.valid_at is not None:
            at = filter.valid_at.isoformat()
            params.append(("valid_from", f"lte.{at}"))
            params.append(("or", f"(valid_until.is.null,valid_until.gt.{at})"))

        # Current-only by default.
        if not filter.include_superseded:
            params.append(("valid_until", "is.null"))

        # Importance bounds.
        if filter.min_importance is not None:
            params.append(("importance", f"gte.{filter.min_importance}"))
        if filter.max_importance is not None:
            params.append(("importance", f"lte.{filter.max_importance}"))

        # Entity references: any-of via jsonb containment of each ref.
        if filter.entity_refs:
            ors = [f'entity_refs.cs.["{ref}"]' for ref in filter.entity_refs]
            params.append(("or", "(" + ",".join(ors) + ")"))

        # Lineage.
        if filter.has_supersedes is not None:
            params.append(("supersedes", "not.is.null" if filter.has_supersedes else "is.null"))
        if filter.has_promoted_from is not None:
            params.append(
                ("promoted_from", "not.is.null" if filter.has_promoted_from else "is.null")
            )

        # Metadata equality filters via jsonb arrow operator.
        for key, value in (filter.metadata_filters or {}).items():
            if not str(key).replace("_", "").isalnum():
                continue  # guard against PostgREST operator injection
            params.append((f"metadata->>{key}", f"eq.{value}"))

        # Ordering.
        order_col = filter.order_by if filter.order_by in _VALID_ORDER_COLUMNS else "created_at"
        params.append(("order", f"{order_col}.{'desc' if filter.order_desc else 'asc'}"))

        # Pagination.
        if filter.limit is not None:
            params.append(("limit", str(filter.limit)))
        if filter.offset:
            params.append(("offset", str(filter.offset)))

        return params

    # -- MemoryStore protocol ----------------------------------------------

    async def save(self, memory: Memory) -> None:
        await self._request(
            "POST",
            f"/{self._table}",
            params=[("on_conflict", "id")],
            json_body=self._memory_to_row(memory),
            prefer="resolution=merge-duplicates,return=minimal",
        )

    async def save_batch(self, memories: list[Memory]) -> None:
        if not memories:
            return
        await self._request(
            "POST",
            f"/{self._table}",
            params=[("on_conflict", "id")],
            json_body=[self._memory_to_row(m) for m in memories],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    async def get(self, memory_id: str) -> Memory | None:
        rows = await self._request(
            "GET",
            f"/{self._table}",
            params=[("id", f"eq.{memory_id}"), ("limit", "1")],
        )
        return self._row_to_memory(rows[0]) if rows else None

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        ids = ",".join(memory_ids)
        rows = await self._request("GET", f"/{self._table}", params=[("id", f"in.({ids})")])
        return [self._row_to_memory(r) for r in (rows or [])]

    async def delete(self, memory_id: str) -> bool:
        rows = await self._request(
            "DELETE",
            f"/{self._table}",
            params=[("id", f"eq.{memory_id}")],
            prefer="return=representation",
        )
        return bool(rows)

    async def delete_batch(self, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        ids = ",".join(memory_ids)
        rows = await self._request(
            "DELETE",
            f"/{self._table}",
            params=[("id", f"in.({ids})")],
            prefer="return=representation",
        )
        return len(rows or [])

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        rows = await self._request(
            "GET", f"/{self._table}", params=self._build_query_params(filter)
        )
        return [self._row_to_memory(r) for r in (rows or [])]

    async def count(self, filter: MemoryFilter) -> int:
        # Reuse the filter but ignore pagination/order; ask PostgREST for an
        # exact count via the Content-Range header instead of fetching rows.
        params = [
            (k, v)
            for (k, v) in self._build_query_params(filter)
            if k not in ("limit", "offset", "order")
        ]
        params.extend([("select", "id"), ("limit", "1")])
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise SupabaseConfigError("httpx is required for the Supabase backend.") from exc
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self._rest}/{self._table}",
                params=params,
                headers=self._headers(prefer="count=exact"),
            )
        resp.raise_for_status()
        content_range = resp.headers.get("content-range", "*/0")
        total = content_range.split("/")[-1]
        return int(total) if total.isdigit() else 0

    async def supersede(
        self,
        old_memory_id: str,
        new_memory: Memory,
        supersede_time: datetime | None = None,
    ) -> Memory:
        when = supersede_time or datetime.utcnow()
        # Close out the old memory and link the lineage in both directions.
        await self._request(
            "PATCH",
            f"/{self._table}",
            params=[("id", f"eq.{old_memory_id}")],
            json_body={"valid_until": _iso(when), "superseded_by": new_memory.id},
            prefer="return=minimal",
        )
        new_memory.supersedes = old_memory_id
        new_memory.valid_from = when
        await self.save(new_memory)
        return new_memory

    async def get_history(self, memory_id: str, include_future: bool = False) -> list[Memory]:
        # Walk backwards through `supersedes` to the chain root, then forward
        # through `superseded_by`, returning oldest-first.
        seen: set[str] = set()
        chain: list[Memory] = []

        async def walk(mid: str | None, follow: str) -> None:
            while mid and mid not in seen:
                mem = await self.get(mid)
                if mem is None:
                    break
                seen.add(mid)
                chain.append(mem)
                mid = getattr(mem, follow)

        start = await self.get(memory_id)
        if start is None:
            return []
        seen.add(start.id)
        chain.append(start)
        await walk(start.supersedes, "supersedes")
        if include_future:
            await walk(start.superseded_by, "superseded_by")
        chain.sort(key=lambda m: m.valid_from)
        return chain

    async def clear_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        params = [("user_id", f"eq.{user_id}")]
        if session_id is not None:
            params.append(("session_id", f"eq.{session_id}"))
        if agent_id is not None:
            params.append(("agent_id", f"eq.{agent_id}"))
        if turn_id is not None:
            params.append(("turn_id", f"eq.{turn_id}"))
        rows = await self._request(
            "DELETE", f"/{self._table}", params=params, prefer="return=representation"
        )
        return len(rows or [])


class SupabaseVectorIndex:
    """A :class:`~headroom.memory.ports.VectorIndex` over the same Supabase table.

    Embeddings live alongside their memories in ``headroom_memories.embedding``,
    so indexing is an upsert and similarity search is the ``match_headroom_memories``
    RPC. This keeps store and index trivially consistent (one row, one source of
    truth) — the opposite of the local split SQLite store + vector file.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._store = SupabaseMemoryStore(config)
        self._dimension = int(getattr(config, "vector_dimension", 384) or 384)

    async def index(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValueError("Cannot index a memory without an embedding")
        await self._store.save(memory)

    async def index_batch(self, memories: list[Memory]) -> int:
        with_emb = [m for m in memories if m.embedding is not None]
        if with_emb:
            await self._store.save_batch(with_emb)
        return len(with_emb)

    async def remove(self, memory_id: str) -> bool:
        # Null out the embedding rather than deleting the memory row.
        await self._store._request(
            "PATCH",
            f"/{self._store._table}",
            params=[("id", f"eq.{memory_id}")],
            json_body={"embedding": None},
            prefer="return=minimal",
        )
        return True

    async def remove_batch(self, memory_ids: list[str]) -> int:
        for mid in memory_ids:
            await self.remove(mid)
        return len(memory_ids)

    async def search(self, filter: VectorFilter) -> list[VectorSearchResult]:
        query_vec = filter.query_vector
        if query_vec is None:
            raise ValueError(
                "SupabaseVectorIndex.search requires a precomputed query_vector; "
                "embed query_text with the configured Embedder first."
            )
        body = {
            "query_embedding": _embedding_to_pg(query_vec),
            "match_count": filter.top_k,
            "min_similarity": filter.min_similarity,
            "p_user_id": filter.user_id,
            "p_session_id": filter.session_id,
            "include_superseded": filter.include_superseded,
        }
        rows = await self._store._request("POST", f"/rpc/{_MATCH_RPC}", json_body=body)
        results: list[VectorSearchResult] = []
        for rank, row in enumerate(rows or [], start=1):
            results.append(
                VectorSearchResult(
                    memory=self._store._row_to_memory(row),
                    similarity=float(row.get("similarity", 0.0)),
                    rank=rank,
                )
            )
        return results

    async def update_embedding(self, memory_id: str, embedding: Any) -> bool:
        await self._store._request(
            "PATCH",
            f"/{self._store._table}",
            params=[("id", f"eq.{memory_id}")],
            json_body={"embedding": _embedding_to_pg(embedding)},
            prefer="return=minimal",
        )
        return True

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        # Live size requires a network round-trip; callers that need it should
        # use SupabaseMemoryStore.count(). Return -1 to signal "unknown".
        return -1
