import random

import pytest

import neuropoker.stats as stats
from neuropoker.stats import compare_evals, convert, discard_equity, evaluate_best


@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    # Ensure deterministic defaults for tests
    monkeypatch.setenv("NEUROPOKER_DISCARD_MAX_SECONDS", "0")
    monkeypatch.setenv("NEUROPOKER_DISCARD_SAMPLES", "5")
    yield


def test_evaluate_best_orders_by_strength():
    board = [convert("Ah"), convert("Kd")]
    hero_rank = evaluate_best(board, [convert("As"), convert("Kh")])
    villain_rank = evaluate_best(board, [convert("2c"), convert("3d")])
    assert compare_evals(hero_rank, villain_rank) == 1


def test_discard_equity_sampling_metadata(monkeypatch):
    # Patch Random to deterministic sampling
    rng = random.Random(0)
    monkeypatch.setattr(stats.random, "Random", lambda: rng)

    hole = [convert("Ah"), convert("7c"), convert("2d")]
    board = [convert("Kd"), convert("Qs")]
    equities, meta = discard_equity(
        hole,
        board,
        n_samples=4,
        max_seconds=0,
        return_metadata=True,
    )
    assert len(equities) == 3
    assert len(meta) == 3
    for entry in meta:
        assert entry["samples"] <= 4
        assert entry["samples"] > 0
    for eq in equities:
        assert 0.0 <= eq <= 1.0


def test_discard_equity_enumeration_fallback(monkeypatch):
    # Force a tiny deck so enumeration runs but stays cheap
    tiny_cards = [convert("4c"), convert("5c"), convert("6c"), convert("7c"), convert("8c")]

    class TinyDeck:
        def __init__(self):
            self.cards = list(tiny_cards)

    monkeypatch.setattr(stats, "Deck", TinyDeck)

    hole = [convert("Ah"), convert("7c"), convert("2d")]
    board = [convert("Kd"), convert("Qs")]
    equities = discard_equity(hole, board, n_samples=0, max_seconds=0)
    assert len(equities) == 3
    for eq in equities:
        assert 0.0 <= eq <= 1.0
