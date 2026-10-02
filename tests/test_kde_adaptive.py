"""Tests for the Phase-4 :class:`shotcloud.kde.AdaptiveKDE`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shotcloud import AdaptiveKDE, CourtGrid
from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.kde._kernel import build_kernel_matrix

# ---------------------------------------------------------------------------
# build_kernel_matrix
# ---------------------------------------------------------------------------


def test_kernel_matrix_columns_sum_to_one() -> None:
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    M = build_kernel_matrix(g, bandwidth=1.5)
    np.testing.assert_allclose(M.sum(axis=0), 1.0, atol=1e-12)


def test_kernel_matrix_strictly_positive() -> None:
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    M = build_kernel_matrix(g, bandwidth=1.5, epsilon=1e-12)
    assert (M > 0).all()


def test_kernel_matrix_shape_is_square() -> None:
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    M = build_kernel_matrix(g, bandwidth=1.5)
    assert M.shape == (g.n_cells, g.n_cells)


def test_kernel_matrix_zero_bandwidth_is_identity() -> None:
    """At bandwidth=0 (no blur), each column k is a near-one-hot at k.

    With ε-floor still applied, the column at k is dominated by index k.
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    M = build_kernel_matrix(g, bandwidth=0.0, epsilon=1e-12)
    # Each column's argmax should be its own index (the source cell).
    assert (np.argmax(M, axis=0) == np.arange(g.n_cells)).all()


# ---------------------------------------------------------------------------
# AdaptiveKDE
# ---------------------------------------------------------------------------


def _synthetic_phase4_df(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(
            {
                "x": float(rng.normal(0, 5)),
                "y": float(rng.normal(15, 5)),
                "player_id": "A" if i % 3 < 2 else "B",
                "opponent": "X" if i % 2 == 0 else "Y",
                "made": int(rng.random() < 0.5),
                "period": (i % 4) + 1,
                "time_remaining_sec": int(60 * (i % 48)),
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)),
            }
        )
    df = pd.DataFrame(rows)
    enc = ContextEncoder.fit(df)
    return df, enc.transform(df)


def _fitted_kde() -> AdaptiveKDE:
    df, ctx = _synthetic_phase4_df(n=120)
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=20, ny=22)
    kde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=30)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
    )
    return kde


def test_adaptive_kde_invariants() -> None:
    kde = _fitted_kde()
    assert kde.is_fitted
    assert kde.context_dim == CONTEXT_DIM
    assert set(kde.players) == {"A", "B"}
    # max_history = 30 caps both players (they each have 40+ shots).
    for pid in kde.players:
        assert kde.n_history[pid] <= 30
        assert kde.cells[pid].shape == (kde.n_history[pid],)
        assert kde.context[pid].shape == (kde.n_history[pid], CONTEXT_DIM)


def test_adaptive_kde_density_with_uniform_relevance_sums_to_one() -> None:
    kde = _fitted_kde()
    for pid in kde.players:
        d = kde.density_with_uniform_relevance(pid)
        np.testing.assert_allclose(d.sum(), 1.0, atol=1e-10)
        assert (d > 0).all()


def test_adaptive_kde_unknown_player_raises() -> None:
    kde = _fitted_kde()
    with pytest.raises(KeyError, match="unknown player"):
        kde.player_history("does-not-exist")


def test_adaptive_kde_invalid_max_history_raises() -> None:
    g = CourtGrid()
    with pytest.raises(ValueError, match="max_history"):
        AdaptiveKDE(grid=g, max_history=0)
    with pytest.raises(ValueError, match="max_history"):
        AdaptiveKDE(grid=g, max_history=-5)


def test_adaptive_kde_unfitted_raises() -> None:
    g = CourtGrid()
    kde = AdaptiveKDE(grid=g)
    assert not kde.is_fitted
    with pytest.raises(KeyError, match="unknown player"):
        kde.player_history("A")


def test_adaptive_kde_drops_backcourt_shots() -> None:
    """Backcourt shots (out-of-court) must not appear in any player's history."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    kde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=None)
    # 10 in-court shots + 5 backcourt shots, all player A.
    df = pd.DataFrame(
        {
            "x": [0.0] * 15,
            "y": [10.0] * 10 + [60.0] * 5,  # last 5 are backcourt
            "player_id": ["A"] * 15,
            "date": ["2024-01-01"] * 15,
            "period": [1] * 15,
            "time_remaining_sec": [0] * 15,
            "opponent": ["X"] * 15,
            "made": [1] * 15,
        }
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
    )
    # All 10 in-court shots survive; 5 backcourt drop.
    assert kde.n_history["A"] == 10


def test_adaptive_kde_subsample_caps_at_max() -> None:
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 5, n),
            "y": rng.normal(10, 5, n),
            "player_id": ["A"] * n,
            "date": [pd.Timestamp("2024-01-01")] * n,
            "period": [1] * n,
            "time_remaining_sec": [0] * n,
            "opponent": ["X"] * n,
            "made": [1] * n,
        }
    )
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(grid=g, bandwidth=1.5, max_history=37, seed=0)
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
    )
    assert kde.n_history["A"] == 37


# ---------------------------------------------------------------------------
# recent_stratified history policy (2026-05-16, q_self readiness work)
# ---------------------------------------------------------------------------


def _build_shots_with_zone_pattern(n_recent: int, n_old: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic per-player shots: ``n_old`` older shots from many zones,
    then ``n_recent`` recent shots concentrated in the rim. Lets us
    verify that recent_stratified keeps recency while still preserving
    older-zone coverage."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    # Older shots: spread across all 8 zones. Pick a representative
    # (x, y) per zone, jittered.
    zone_anchors = [
        (0.0, 2.0),  # RA
        (4.0, 8.0),  # paint
        (10.0, 12.0),  # mid
        (-23.0, 3.0),  # corner3-L
        (23.0, 3.0),  # corner3-R
        (-15.0, 22.0),  # wing3-L
        (15.0, 22.0),  # wing3-R
        (0.0, 25.0),  # topkey3
    ]
    rows = []
    for i in range(n_old):
        zx, zy = zone_anchors[i % len(zone_anchors)]
        rows.append(
            {
                "x": float(zx + rng.normal(0, 0.5)),
                "y": float(zy + rng.normal(0, 0.5)),
                "player_id": "A",
                "date": base_date + pd.Timedelta(days=i),
            }
        )
    # Recent shots: rim cluster only (zone 0).
    for j in range(n_recent):
        rows.append(
            {
                "x": float(0.0 + rng.normal(0, 0.5)),
                "y": float(2.0 + rng.normal(0, 0.5)),
                "player_id": "A",
                "date": base_date + pd.Timedelta(days=n_old + j),
            }
        )
    df = pd.DataFrame(rows)
    df["period"] = 1
    df["time_remaining_sec"] = 0
    df["opponent"] = "X"
    df["made"] = 1
    return df


def test_recent_stratified_keeps_exactly_max_history() -> None:
    """Capped history has exactly max_history unique shots."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=400, n_old=600)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(
        grid=g,
        bandwidth=1.5,
        max_history=200,
        seed=0,
        history_policy="recent_stratified",
        recent_history=140,
        stratified_history=60,
    )
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    assert kde.n_history["A"] == 200


def test_recent_stratified_always_includes_most_recent_shots() -> None:
    """The last ``recent_history`` chronological shots must always be kept."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=400, n_old=600)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(
        grid=g,
        bandwidth=1.5,
        max_history=200,
        seed=0,
        history_policy="recent_stratified",
        recent_history=140,
        stratified_history=60,
    )
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    kept_dates = set(int(d) for d in kde.dates["A"])
    # The last 140 chronological dates are at the end of base_date + 600..999.
    expected_recent = set(
        int(np.datetime64(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "D").astype(np.int64))
        for i in range(1000 - 140, 1000)
    )
    assert expected_recent.issubset(kept_dates)


def test_recent_stratified_aligns_cells_context_coords_dates() -> None:
    """All four per-shot arrays stay aligned after subsampling."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=400, n_old=600)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(
        grid=g,
        bandwidth=1.5,
        max_history=200,
        seed=0,
        history_policy="recent_stratified",
        recent_history=140,
        stratified_history=60,
    )
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    n = kde.n_history["A"]
    assert kde.cells["A"].shape == (n,)
    assert kde.context["A"].shape == (n, ctx.shape[1])
    assert kde.coords["A"].shape == (n, 2)
    assert kde.dates["A"].shape == (n,)
    # Per-shot consistency: each stored coord should reconstruct the
    # stored cell index via coord_to_cell.
    recomputed_cells = g.coord_to_cell(
        kde.coords["A"][:, 0].astype(np.float64),
        kde.coords["A"][:, 1].astype(np.float64),
    )
    np.testing.assert_array_equal(recomputed_cells, kde.cells["A"])


def test_recent_stratified_preserves_older_zone_coverage() -> None:
    """Stratified portion should sample from older zones (not just rim).

    The synthetic data has 8 zones in the older pool. After subsampling,
    the stored support should include shots from at least 6 of those
    older zones — confirming the stratified sample actually spreads
    across zones rather than collapsing to the dominant zone.
    """
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=400, n_old=600)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(
        grid=g,
        bandwidth=1.5,
        max_history=200,
        seed=0,
        history_policy="recent_stratified",
        recent_history=140,
        stratified_history=60,
    )
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    from shotcloud.data.zones import zone_from_xy_vectorized

    kept_zones = zone_from_xy_vectorized(
        kde.coords["A"][:, 0].astype(np.float64),
        kde.coords["A"][:, 1].astype(np.float64),
    )
    unique_kept_zones = set(int(z) for z in np.unique(kept_zones) if z >= 0)
    # All 8 zones exist in the data; stratified should keep most of them.
    assert len(unique_kept_zones) >= 6


def test_recent_stratified_is_deterministic_under_seed() -> None:
    """Two fits with the same seed should produce identical histories."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=400, n_old=600)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)

    def _fit() -> AdaptiveKDE:
        kde = AdaptiveKDE(
            grid=g,
            bandwidth=1.5,
            max_history=200,
            seed=42,
            history_policy="recent_stratified",
            recent_history=140,
            stratified_history=60,
        )
        kde.fit(
            x=df["x"].to_numpy(),
            y=df["y"].to_numpy(),
            player_id=df["player_id"].to_numpy(),
            context_features=ctx,
            date=df["date"].to_numpy(),
        )
        return kde

    kde_a = _fit()
    kde_b = _fit()
    np.testing.assert_array_equal(kde_a.cells["A"], kde_b.cells["A"])
    np.testing.assert_array_equal(kde_a.dates["A"], kde_b.dates["A"])
    np.testing.assert_array_equal(kde_a.coords["A"], kde_b.coords["A"])


def test_recent_stratified_no_subsample_when_under_cap() -> None:
    """If player has fewer shots than max_history, keep them all."""
    g = CourtGrid(xlim=(-25.0, 25.0), ylim=(-5.0, 47.0), nx=10, ny=11)
    df = _build_shots_with_zone_pattern(n_recent=50, n_old=80)
    enc = ContextEncoder.fit(df)
    ctx = enc.transform(df)
    kde = AdaptiveKDE(
        grid=g,
        bandwidth=1.5,
        max_history=200,
        seed=0,
        history_policy="recent_stratified",
        recent_history=140,
        stratified_history=60,
    )
    kde.fit(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        player_id=df["player_id"].to_numpy(),
        context_features=ctx,
        date=df["date"].to_numpy(),
    )
    # 50 + 80 = 130 ≤ 200, so all should be kept.
    assert kde.n_history["A"] == 130


def test_history_policy_rejects_mismatched_sums() -> None:
    """recent_history + stratified_history must equal max_history."""
    g = CourtGrid()
    with pytest.raises(ValueError, match=r"recent_history.*stratified_history"):
        AdaptiveKDE(
            grid=g,
            history_policy="recent_stratified",
            max_history=500,
            recent_history=350,
            stratified_history=100,  # 350 + 100 ≠ 500
        )


def test_history_policy_rejects_unknown_policy() -> None:
    g = CourtGrid()
    with pytest.raises(ValueError, match="history_policy"):
        AdaptiveKDE(grid=g, history_policy="something_else")
