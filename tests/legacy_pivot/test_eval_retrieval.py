"""Tests for :func:`shotcloud.evaluation.top_k_retrieval`."""

from __future__ import annotations

import numpy as np
import pytest

from shotcloud.legacy_pivot.eval_retrieval import RetrievalResult, top_k_retrieval


def _well_separated_clouds(n_per_player: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Three players whose shot clouds are far apart in 2D → easy retrieval."""
    rng = np.random.default_rng(seed)
    centers = {"A": (-10.0, 5.0), "B": (10.0, 5.0), "C": (0.0, 25.0)}
    return {
        pid: np.stack(
            [rng.normal(cx, 1.0, n_per_player), rng.normal(cy, 1.0, n_per_player)],
            axis=1,
        )
        for pid, (cx, cy) in centers.items()
    }


def _identical_clouds(n_per_player: int = 100) -> dict[str, np.ndarray]:
    """Three players whose generated clouds are all identical → near-chance retrieval."""
    rng = np.random.default_rng(0)
    same = np.stack([rng.normal(0, 5, n_per_player), rng.normal(15, 5, n_per_player)], axis=1)
    return {"A": same, "B": same.copy(), "C": same.copy()}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def test_returns_retrieval_result_with_expected_fields() -> None:
    real = _well_separated_clouds(seed=0)
    gen = _well_separated_clouds(seed=1)
    result = top_k_retrieval(real, gen, k=2, n_projections=30, seed=0)

    assert isinstance(result, RetrievalResult)
    assert result.n_players == 3
    assert result.k == 2
    assert set(result.ranks.keys()) == {"A", "B", "C"}
    for r in result.ranks.values():
        assert 1 <= r <= 3


# ---------------------------------------------------------------------------
# Correctness on well-separated vs collapsed clouds
# ---------------------------------------------------------------------------


def test_well_separated_clouds_yield_perfect_top1() -> None:
    """When clouds cluster in clearly distinct regions, top-1 == 100%."""
    real = _well_separated_clouds(seed=0)
    gen = _well_separated_clouds(seed=1)
    result = top_k_retrieval(real, gen, k=1, n_projections=50, seed=0)
    assert result.top1_accuracy == 1.0
    assert result.mean_rank == 1.0


def test_collapsed_generation_scores_near_chance() -> None:
    """When all generated clouds are identical, retrieval is essentially a tie."""
    real = _well_separated_clouds(seed=0)
    gen = _identical_clouds()
    result = top_k_retrieval(real, gen, k=1, n_projections=50, seed=0)
    # All gen clouds equidistant → ranks driven by argsort tie-breaking.
    # We just verify it isn't perfect (model isn't recovering identity).
    assert result.top1_accuracy < 1.0


def test_top1_accuracy_le_topk_accuracy() -> None:
    real = _well_separated_clouds(seed=0)
    gen = _well_separated_clouds(seed=1)
    result = top_k_retrieval(real, gen, k=2, n_projections=30, seed=0)
    assert result.top1_accuracy <= result.topk_accuracy


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_mismatched_keys_raises() -> None:
    real = _well_separated_clouds()
    gen = _well_separated_clouds()
    del gen["A"]
    with pytest.raises(ValueError, match="same key set"):
        top_k_retrieval(real, gen)


def test_too_few_players_raises() -> None:
    p = np.zeros((10, 2))
    with pytest.raises(ValueError, match="at least 2 players"):
        top_k_retrieval({"A": p}, {"A": p})


def test_invalid_k_raises() -> None:
    real = _well_separated_clouds()
    gen = _well_separated_clouds()
    with pytest.raises(ValueError, match="k=0"):
        top_k_retrieval(real, gen, k=0)
    with pytest.raises(ValueError, match="k=99"):
        top_k_retrieval(real, gen, k=99)


# ---------------------------------------------------------------------------
# Custom distance
# ---------------------------------------------------------------------------


def test_custom_distance_function() -> None:
    """Caller can override the distance function (e.g., to use a fake one for tests)."""
    real = _well_separated_clouds()
    gen = _well_separated_clouds()
    n_calls = [0]

    def fake_distance(p: np.ndarray, q: np.ndarray) -> float:
        n_calls[0] += 1
        return float(np.abs(p.mean(axis=0) - q.mean(axis=0)).sum())

    result = top_k_retrieval(real, gen, k=1, distance=fake_distance)
    assert result.n_players == 3
    assert n_calls[0] == 9  # 3x3 distance matrix
