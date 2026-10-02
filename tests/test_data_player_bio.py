"""Tests for :mod:`shotcloud.data.player_bio` loader and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shotcloud.data.player_bio import (
    POSITION_GROUPS,
    compute_age_years,
    load_player_bio,
    position_group_onehot_matrix,
    position_group_to_onehot,
)


def _make_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    p = tmp_path / "player_bio.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _ok_row(player_id: int, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "player_id": player_id,
        "display_name": f"Player {player_id}",
        "birthdate": "1990-01-15",
        "height_inches": 78,
        "weight_lbs": 215,
        "position_raw": "Guard-Forward",
        "position_group": "SG",
        "source": "nba_stats:CommonPlayerInfo",
        "fetched_at": "2026-05-16T22:00:00+00:00",
        "status": "ok",
        "error_message": None,
    }
    base.update(overrides)
    return base


def test_load_player_bio_types(tmp_path: Path) -> None:
    """Loaded dataframe has the dtypes downstream code expects."""
    p = _make_csv(tmp_path, [_ok_row(101), _ok_row(102, height_inches=72)])
    df = load_player_bio(p)
    assert df["player_id"].dtype == np.dtype("int64")
    # birthdate must be datetime64
    assert pd.api.types.is_datetime64_any_dtype(df["birthdate"])
    # nullable Int64 for the integer-with-missing columns
    assert str(df["height_inches"].dtype) == "Int64"
    assert str(df["weight_lbs"].dtype) == "Int64"
    # string dtype for the categorical / text columns
    assert str(df["display_name"].dtype) == "string"
    assert str(df["position_raw"].dtype) == "string"
    assert str(df["position_group"].dtype) == "string"
    assert str(df["status"].dtype) == "string"


def test_load_player_bio_keeps_missing_status_rows(tmp_path: Path) -> None:
    """Failed / missing rows are preserved (caller decides how to handle)."""
    rows = [
        _ok_row(101),
        _ok_row(
            102,
            display_name=None,
            birthdate=None,
            height_inches=None,
            weight_lbs=None,
            position_raw=None,
            position_group=None,
            status="failed",
            error_message="HTTP 503",
        ),
    ]
    p = _make_csv(tmp_path, rows)
    df = load_player_bio(p)
    assert len(df) == 2
    failed = df[df["status"] == "failed"].iloc[0]
    assert pd.isna(failed["height_inches"])
    assert pd.isna(failed["birthdate"])


def test_load_player_bio_rejects_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame([{"player_id": 1, "wrong_col": 2}])
    p = tmp_path / "broken.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_player_bio(p)


def test_load_player_bio_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_player_bio(tmp_path / "nonexistent.csv")


def test_compute_age_years_known_cases(tmp_path: Path) -> None:
    """Known birthdate → age at ref_date matches manual computation."""
    p = _make_csv(
        tmp_path,
        [
            _ok_row(1, birthdate="1990-01-15"),
            _ok_row(2, birthdate="2000-12-31"),
            _ok_row(3, birthdate=None),  # NaT
        ],
    )
    df = load_player_bio(p)
    ages = compute_age_years(df["birthdate"], "2024-01-15")
    # Player 1: born 1990-01-15 → 34.0 years on 2024-01-15
    np.testing.assert_allclose(ages[0], 34.0, atol=0.01)
    # Player 2: born 2000-12-31 → ~23.04 years on 2024-01-15
    np.testing.assert_allclose(ages[1], 23.04, atol=0.05)
    # Player 3: NaT birthdate → NaN age
    assert np.isnan(ages[2])


def test_position_group_to_onehot_known() -> None:
    pg_onehot = position_group_to_onehot("PG")
    np.testing.assert_array_equal(pg_onehot, [1, 0, 0, 0, 0])
    c_onehot = position_group_to_onehot("C")
    np.testing.assert_array_equal(c_onehot, [0, 0, 0, 0, 1])


def test_position_group_to_onehot_unknown_is_all_zero() -> None:
    assert position_group_to_onehot(None).sum() == 0
    assert position_group_to_onehot("bogus").sum() == 0
    assert position_group_to_onehot(float("nan")).sum() == 0


def test_position_group_onehot_matrix_shape_and_content(tmp_path: Path) -> None:
    p = _make_csv(
        tmp_path,
        [
            _ok_row(1, position_group="PG"),
            _ok_row(2, position_group="SF"),
            _ok_row(3, position_group=None),
        ],
    )
    df = load_player_bio(p)
    matrix = position_group_onehot_matrix(df["position_group"])
    assert matrix.shape == (3, len(POSITION_GROUPS))
    np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0, 0])  # PG
    np.testing.assert_array_equal(matrix[1], [0, 0, 1, 0, 0])  # SF
    np.testing.assert_array_equal(matrix[2], [0, 0, 0, 0, 0])  # null
