"""Tests for :mod:`shotcloud.data.positions`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shotcloud import load_shots
from shotcloud.data.positions import (
    POSITION_GROUPS,
    assign_position_group,
    derive_positions_from_ra_rate,
    ra_rate_per_player,
)

SHOT_FLOW_CSV = Path("/Users/aarondanielson/Dropbox/shot_flow/data/shot_data.csv")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_position_groups_contain_three_categories() -> None:
    assert set(POSITION_GROUPS) == {"big", "wing", "guard"}


# ---------------------------------------------------------------------------
# assign_position_group — scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ra_rate, expected",
    [
        (0.50, "big"),
        (0.35, "big"),  # boundary inclusive
        (0.30, "wing"),
        (0.20, "wing"),  # boundary inclusive
        (0.15, "guard"),
        (0.0, "guard"),
    ],
)
def test_assign_position_group_thresholds(ra_rate: float, expected: str) -> None:
    assert assign_position_group(ra_rate) == expected


def test_assign_position_group_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        assign_position_group(0.3, big_threshold=0.2, wing_threshold=0.4)
    with pytest.raises(ValueError, match="thresholds"):
        assign_position_group(0.3, big_threshold=1.5)


def test_assign_position_group_custom_thresholds() -> None:
    """Caller can shift the boundaries."""
    assert assign_position_group(0.45, big_threshold=0.50) == "wing"
    assert assign_position_group(0.55, big_threshold=0.50) == "big"


# ---------------------------------------------------------------------------
# ra_rate_per_player
# ---------------------------------------------------------------------------


def test_ra_rate_per_player_basic() -> None:
    """A player taking only RA shots gets rate 1.0; only-3PT gets 0.0."""
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, 1.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 25.0, 27.0],
            "player_id": ["RIM", "RIM", "RIM", "ARC", "ARC"],
        }
    )
    rates = ra_rate_per_player(df)
    assert rates["RIM"] == pytest.approx(1.0)
    assert rates["ARC"] == pytest.approx(0.0)


def test_ra_rate_per_player_dropna() -> None:
    df = pd.DataFrame(
        {
            "x": [0.0, np.nan, 1.0],
            "y": [0.0, 5.0, 0.0],
            "player_id": ["A", "A", "A"],
        }
    )
    rates = ra_rate_per_player(df)
    # Only the two non-NaN shots are used; both within 4ft of basket → 1.0.
    assert rates["A"] == pytest.approx(1.0)


def test_ra_rate_per_player_missing_column_raises() -> None:
    df = pd.DataFrame({"x": [0.0], "y": [0.0]})
    with pytest.raises(KeyError, match="player_id"):
        ra_rate_per_player(df)


def test_ra_rate_per_player_empty_returns_empty_series() -> None:
    df = pd.DataFrame({"x": [], "y": [], "player_id": []})
    rates = ra_rate_per_player(df)
    assert len(rates) == 0


# ---------------------------------------------------------------------------
# derive_positions_from_ra_rate — labeling
# ---------------------------------------------------------------------------


def _synthetic_three_archetypes(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Three players: a Big (paint-heavy), a Wing (mixed), and a Guard (mostly mid+arc)."""
    rng = np.random.default_rng(seed)
    rows = []

    # Big: 50% RA, 30% paint, 20% midrange.
    big_x = np.concatenate(
        [
            rng.normal(0.0, 1.0, n // 2),  # RA
            rng.normal(2.0, 2.0, int(n * 0.3)),  # paint
            rng.normal(0.0, 5.0, int(n * 0.2)),  # midrange
        ]
    )
    big_y = np.concatenate(
        [
            rng.uniform(-1.0, 3.5, n // 2),  # within 4 ft
            rng.uniform(5.0, 12.0, int(n * 0.3)),
            rng.uniform(15.0, 18.0, int(n * 0.2)),
        ]
    )
    rows.extend({"x": x, "y": y, "player_id": "BIG"} for x, y in zip(big_x, big_y, strict=True))

    # Wing: 25% RA, 35% midrange, 40% threes.
    wing_x = np.concatenate(
        [
            rng.normal(0.0, 1.0, int(n * 0.25)),
            rng.normal(0.0, 5.0, int(n * 0.35)),
            rng.uniform(-22.0, 22.0, int(n * 0.40)),
        ]
    )
    wing_y = np.concatenate(
        [
            rng.uniform(-1.0, 3.5, int(n * 0.25)),
            rng.uniform(10.0, 18.0, int(n * 0.35)),
            rng.uniform(22.0, 26.0, int(n * 0.40)),
        ]
    )
    rows.extend({"x": x, "y": y, "player_id": "WING"} for x, y in zip(wing_x, wing_y, strict=True))

    # Guard: 10% RA, 30% midrange, 60% threes.
    guard_x = np.concatenate(
        [
            rng.normal(0.0, 1.0, int(n * 0.10)),
            rng.normal(0.0, 6.0, int(n * 0.30)),
            rng.uniform(-23.0, 23.0, int(n * 0.60)),
        ]
    )
    guard_y = np.concatenate(
        [
            rng.uniform(-1.0, 3.5, int(n * 0.10)),
            rng.uniform(12.0, 19.0, int(n * 0.30)),
            rng.uniform(23.0, 27.0, int(n * 0.60)),
        ]
    )
    rows.extend(
        {"x": x, "y": y, "player_id": "GUARD"} for x, y in zip(guard_x, guard_y, strict=True)
    )

    return pd.DataFrame(rows)


def test_derive_positions_three_archetypes() -> None:
    df = _synthetic_three_archetypes()
    positions = derive_positions_from_ra_rate(df)
    assert positions["BIG"] == "big"
    assert positions["WING"] == "wing"
    assert positions["GUARD"] == "guard"


def test_derive_positions_sparse_player_gets_fallback() -> None:
    """A player with < min_shots gets the fallback label."""
    df = pd.DataFrame(
        {
            "x": [0.0, 0.0, 0.0],  # 3 RA shots
            "y": [0.0, 0.0, 0.0],
            "player_id": ["SPARSE"] * 3,
        }
    )
    positions = derive_positions_from_ra_rate(df, min_shots=10)
    # Despite having 100% RA rate, fewer than 10 shots → fallback.
    assert positions["SPARSE"] == "wing"


def test_derive_positions_keys_match_unique_player_ids() -> None:
    """Output dict keys cover exactly the players present in the input."""
    df = _synthetic_three_archetypes()
    positions = derive_positions_from_ra_rate(df)
    assert set(positions.keys()) == set(df["player_id"].unique())


def test_derive_positions_handles_integer_player_ids() -> None:
    """Integer IDs should work — pandas/numpy may coerce types in groupby."""
    df = _synthetic_three_archetypes()
    df["player_id"] = pd.Categorical(df["player_id"]).codes  # to ints
    positions = derive_positions_from_ra_rate(df)
    assert len(positions) == 3
    assert set(positions.values()).issubset(set(POSITION_GROUPS))


def test_derive_positions_invalid_min_shots_raises() -> None:
    df = _synthetic_three_archetypes()
    with pytest.raises(ValueError, match="min_shots"):
        derive_positions_from_ra_rate(df, min_shots=0)


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SHOT_FLOW_CSV.exists(), reason="shot_flow CSV not available")
def test_derive_positions_on_real_nba_data() -> None:
    """On real NBA data, we should see a reasonable mix of bigs/wings/guards."""
    df = load_shots(SHOT_FLOW_CSV, nrows=50_000)
    # Restrict to players with enough shots for stable inference.
    counts = df["player_id"].value_counts()
    eligible = counts[counts >= 50].index
    df = df[df["player_id"].isin(eligible)].copy()

    positions = derive_positions_from_ra_rate(df)
    from collections import Counter

    counts = Counter(positions.values())
    # Every group should appear, and no group should completely dominate.
    assert counts.get("big", 0) > 0
    assert counts.get("wing", 0) > 0
    assert counts.get("guard", 0) > 0
    n_total = sum(counts.values())
    for group, n in counts.items():
        frac = n / n_total
        assert 0.05 < frac < 0.85, (
            f"position {group!r} has implausible share {frac:.2%} of {n_total} players"
        )
