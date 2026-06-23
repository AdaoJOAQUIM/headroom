"""Tests for the Supabase memory backend (headroom.memory.adapters.supabase_store).

These pin the pure, network-free logic — credential resolution, Memory <-> row
serialization, embedding <-> pgvector conversion, and MemoryFilter -> PostgREST
parameter translation — so the adapter stays faithful to the SQLite reference
without requiring a live Supabase project.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from headroom.memory.adapters.supabase_store import (
    SupabaseConfigError,
    SupabaseMemoryStore,
    _embedding_from_pg,
    _embedding_to_pg,
    _resolve_credentials,
)
from headroom.memory.models import Memory, ScopeLevel
from headroom.memory.ports import MemoryFilter

np = pytest.importorskip("numpy")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> SupabaseMemoryStore:
    monkeypatch.setenv("HEADROOM_SUPABASE_URL", "https://proj.supabase.co/")
    monkeypatch.setenv("HEADROOM_SUPABASE_KEY", "anon-key")
    monkeypatch.delenv("HEADROOM_SUPABASE_MEMORY_TABLE", raising=False)
    return SupabaseMemoryStore()


# --------------------------------------------------------------------------- #
# Credential resolution
# --------------------------------------------------------------------------- #


def test_resolve_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_SUPABASE_URL", "https://x.supabase.co/")
    monkeypatch.setenv("HEADROOM_SUPABASE_KEY", "k")
    monkeypatch.setenv("HEADROOM_SUPABASE_MEMORY_TABLE", "custom_mem")

    url, key, table = _resolve_credentials(None)

    assert url == "https://x.supabase.co"  # trailing slash stripped
    assert key == "k"
    assert table == "custom_mem"


def test_resolve_credentials_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_SUPABASE_URL", raising=False)
    monkeypatch.delenv("HEADROOM_SUPABASE_KEY", raising=False)

    with pytest.raises(SupabaseConfigError):
        _resolve_credentials(None)


def test_resolve_credentials_default_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("HEADROOM_SUPABASE_KEY", "k")
    monkeypatch.delenv("HEADROOM_SUPABASE_MEMORY_TABLE", raising=False)

    _, _, table = _resolve_credentials(None)
    assert table == "headroom_memories"


# --------------------------------------------------------------------------- #
# Embedding conversion
# --------------------------------------------------------------------------- #


def test_embedding_roundtrip() -> None:
    vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    literal = _embedding_to_pg(vec)
    assert isinstance(literal, str)
    assert literal.startswith("[") and literal.endswith("]")

    restored = _embedding_from_pg(literal)
    assert restored is not None
    assert np.allclose(restored, vec, atol=1e-6)


def test_embedding_none_passthrough() -> None:
    assert _embedding_to_pg(None) is None
    assert _embedding_from_pg(None) is None


def test_embedding_from_pg_accepts_list() -> None:
    restored = _embedding_from_pg([1.0, 2.0])
    assert np.allclose(restored, np.array([1.0, 2.0], dtype=np.float32))


# --------------------------------------------------------------------------- #
# Serialization round-trip
# --------------------------------------------------------------------------- #


def test_memory_row_roundtrip(store: SupabaseMemoryStore) -> None:
    mem = Memory(
        id="m1",
        content="user prefers dark mode",
        user_id="alice",
        session_id="s1",
        importance=0.8,
        entity_refs=["dark_mode", "ui"],
        promotion_chain=["c0"],
        metadata={"project": "headroom", "source": "extraction"},
        embedding=np.array([0.5, 0.25], dtype=np.float32),
    )

    row = store._memory_to_row(mem)

    # JSON-native columns stay Python objects (not json.dumps'd strings).
    assert row["entity_refs"] == ["dark_mode", "ui"]
    assert row["metadata"]["project"] == "headroom"
    assert row["promotion_chain"] == ["c0"]
    assert isinstance(row["embedding"], str)
    assert row["created_at"] is not None

    back = store._row_to_memory(row)
    assert back.id == "m1"
    assert back.content == "user prefers dark mode"
    assert back.user_id == "alice"
    assert back.session_id == "s1"
    assert back.importance == pytest.approx(0.8)
    assert back.entity_refs == ["dark_mode", "ui"]
    assert back.metadata == {"project": "headroom", "source": "extraction"}
    assert np.allclose(back.embedding, mem.embedding, atol=1e-6)


def test_row_to_memory_tolerates_missing_optionals(store: SupabaseMemoryStore) -> None:
    row = {
        "id": "m2",
        "content": "x",
        "user_id": "bob",
        "created_at": "2026-06-23T10:00:00+00:00",
        "valid_from": "2026-06-23T10:00:00+00:00",
    }
    mem = store._row_to_memory(row)
    assert mem.id == "m2"
    assert mem.entity_refs == []
    assert mem.metadata == {}
    assert mem.embedding is None
    assert mem.scope_level == ScopeLevel.USER


# --------------------------------------------------------------------------- #
# Filter -> PostgREST params
# --------------------------------------------------------------------------- #


def _params(store: SupabaseMemoryStore, flt: MemoryFilter) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in store._build_query_params(flt):
        out.setdefault(k, []).append(v)
    return out


def test_query_params_scope_and_default_current(store: SupabaseMemoryStore) -> None:
    params = _params(store, MemoryFilter(user_id="alice", session_id="s1"))
    assert params["user_id"] == ["eq.alice"]
    assert params["session_id"] == ["eq.s1"]
    # Current-only is the default (not include_superseded).
    assert params["valid_until"] == ["is.null"]
    assert params["order"] == ["created_at.desc"]


def test_query_params_include_superseded_drops_valid_until(
    store: SupabaseMemoryStore,
) -> None:
    params = _params(store, MemoryFilter(user_id="a", include_superseded=True))
    assert "valid_until" not in params


def test_query_params_importance_and_pagination(store: SupabaseMemoryStore) -> None:
    params = _params(
        store,
        MemoryFilter(user_id="a", min_importance=0.5, limit=20, offset=40),
    )
    assert params["importance"] == ["gte.0.5"]
    assert params["limit"] == ["20"]
    assert params["offset"] == ["40"]


def test_query_params_temporal(store: SupabaseMemoryStore) -> None:
    after = datetime(2026, 1, 1)
    params = _params(store, MemoryFilter(user_id="a", created_after=after))
    assert params["created_at"] == [f"gte.{after.isoformat()}"]


def test_query_params_scope_levels_uses_or_group(store: SupabaseMemoryStore) -> None:
    params = _params(store, MemoryFilter(scope_levels=[ScopeLevel.USER, ScopeLevel.TURN]))
    assert "or" in params
    or_clause = params["or"][0]
    assert "session_id.is.null" in or_clause
    assert "turn_id.not.is.null" in or_clause


def test_query_params_metadata_filter_rejects_injection(
    store: SupabaseMemoryStore,
) -> None:
    params = _params(
        store,
        MemoryFilter(
            user_id="a",
            metadata_filters={"project": "headroom", "bad key) or 1=1--": "x"},
        ),
    )
    # Legitimate key passes through; the injection-y key is silently skipped.
    assert params["metadata->>project"] == ["eq.headroom"]
    assert not any(k.startswith("metadata->>bad") for k in params)


def test_query_params_invalid_order_falls_back(store: SupabaseMemoryStore) -> None:
    params = _params(store, MemoryFilter(user_id="a", order_by="; DROP TABLE"))
    assert params["order"] == ["created_at.desc"]
