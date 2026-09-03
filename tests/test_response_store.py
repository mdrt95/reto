"""Unit contract for the bounded ``/v1/responses`` continuity state store (issue #27).

The store carries only :class:`ConversationState` — catalog IDs and enum values —
never message or answer text. These tests pin its bounds: TTL expiry, capacity
eviction, fail-closed misses, and safe concurrent access.
"""

import threading

import pytest

from src.agent.contracts import ConversationState
from src.protocol.response_store import ResponseStateStore


class FakeClock:
    """A monotonic clock the test advances by hand for deterministic TTL checks."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _state(topic: str = "projects") -> ConversationState:
    return ConversationState(last_topic=topic, response_language="en")


def test_put_then_get_returns_the_same_state() -> None:
    store = ResponseStateStore(ttl_seconds=60, max_entries=8)
    state = _state()

    store.put("resp_a", state)

    assert store.get("resp_a") == state


def test_get_unknown_id_returns_none() -> None:
    store = ResponseStateStore(ttl_seconds=60, max_entries=8)

    assert store.get("resp_missing") is None


def test_get_after_ttl_returns_none_and_drops_the_entry() -> None:
    clock = FakeClock()
    store = ResponseStateStore(ttl_seconds=30, max_entries=8, clock=clock)
    store.put("resp_a", _state())

    clock.advance(30)

    assert store.get("resp_a") is None
    assert len(store) == 0


def test_entry_just_before_ttl_still_resolves() -> None:
    clock = FakeClock()
    store = ResponseStateStore(ttl_seconds=30, max_entries=8, clock=clock)
    store.put("resp_a", _state())

    clock.advance(29.999)

    assert store.get("resp_a") is not None


def test_capacity_overflow_evicts_the_oldest_entry_first() -> None:
    store = ResponseStateStore(ttl_seconds=60, max_entries=2)

    store.put("resp_1", _state("projects"))
    store.put("resp_2", _state("skills"))
    store.put("resp_3", _state("experience"))

    assert store.get("resp_1") is None
    assert store.get("resp_2") is not None
    assert store.get("resp_3") is not None
    assert len(store) == 2


def test_re_put_refreshes_recency_so_the_refreshed_entry_survives() -> None:
    store = ResponseStateStore(ttl_seconds=60, max_entries=2)

    store.put("resp_1", _state())
    store.put("resp_2", _state())
    store.put("resp_1", _state("skills"))  # refresh: resp_2 is now the oldest
    store.put("resp_3", _state())

    assert store.get("resp_2") is None
    assert store.get("resp_1") is not None
    assert store.get("resp_3") is not None


def test_expired_entries_are_pruned_on_put() -> None:
    clock = FakeClock()
    store = ResponseStateStore(ttl_seconds=10, max_entries=8, clock=clock)
    store.put("resp_old", _state())

    clock.advance(11)
    store.put("resp_new", _state())

    assert len(store) == 1
    assert store.get("resp_old") is None


@pytest.mark.parametrize("bad", [0, -1])
def test_constructor_rejects_a_nonpositive_ttl(bad: int) -> None:
    with pytest.raises(ValueError):
        ResponseStateStore(ttl_seconds=bad, max_entries=8)


@pytest.mark.parametrize("bad", [0, -1])
def test_constructor_rejects_a_nonpositive_capacity(bad: int) -> None:
    with pytest.raises(ValueError):
        ResponseStateStore(ttl_seconds=60, max_entries=bad)


def test_concurrent_put_and_get_stay_bounded_and_raise_nothing() -> None:
    store = ResponseStateStore(ttl_seconds=60, max_entries=50)
    errors: list[BaseException] = []

    def worker(base: int) -> None:
        try:
            for i in range(200):
                response_id = f"resp_{base}_{i}"
                store.put(response_id, _state())
                store.get(response_id)
                store.get("resp_absent")
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store) <= 50


def test_stored_state_never_carries_a_free_text_field() -> None:
    """Belt-and-suspenders: nothing resembling prose can round-trip through the store."""
    store = ResponseStateStore(ttl_seconds=60, max_entries=8)
    state = ConversationState(
        last_topic="projects",
        last_source_ids=["project:proj-sybil"],
        response_language="es",
        delivered_fact_ids=["fact:project:proj-sybil"],
    )

    store.put("resp_a", state)
    restored = store.get("resp_a")

    assert restored is not None
    for value in restored.model_dump().values():
        assert not (isinstance(value, str) and " " in value)
        if isinstance(value, list):
            assert all(not (isinstance(item, str) and " " in item) for item in value)
