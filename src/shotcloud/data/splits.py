"""Train / val / test split utilities for shot tables.

Two split modes (per [plan.md §10 decision 5](../../docs/plan.md)):

- :func:`split_by_season` — partition by NBA season string (preferred for
  paper reproducibility).
- :func:`split_fractional` — random fractional split with a fixed seed
  (useful for quick iteration on a single-season slice).

Both return three DataFrames in train / val / test order.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def split_by_season(
    df: pd.DataFrame,
    *,
    train_seasons: Iterable[str],
    val_seasons: Iterable[str],
    test_seasons: Iterable[str],
    season_column: str = "season",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition ``df`` into train / val / test by ``season_column``.

    Seasons must be disjoint across splits. Rows whose season is not in
    any of the three sets are dropped (with a warning if many).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``season_column``.
    train_seasons, val_seasons, test_seasons : iterable of str
        Season identifiers (e.g., ``"2023-24"``).
    season_column : str, default ``"season"``
    """
    if season_column not in df.columns:
        raise KeyError(
            f"split_by_season requires column {season_column!r}; "
            f"available columns: {list(df.columns)[:20]}"
        )

    train_set = set(train_seasons)
    val_set = set(val_seasons)
    test_set = set(test_seasons)

    overlap = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlap:
        raise ValueError(f"split sets overlap on seasons: {sorted(overlap)}")

    season = df[season_column]
    train = df.loc[season.isin(train_set)].reset_index(drop=True)
    val = df.loc[season.isin(val_set)].reset_index(drop=True)
    test = df.loc[season.isin(test_set)].reset_index(drop=True)
    return train, val, test


def split_fractional(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random fractional split.

    Parameters
    ----------
    df : pd.DataFrame
    train_frac, val_frac, test_frac : float
        Must sum to 1.0 (within float tolerance).
    seed : int, default 42
    group_column : str, optional
        If given, splits at the group level so all rows of a given group
        end up in the same split (e.g., ``group_column="game_id"`` keeps
        a game's shots together). If ``None``, splits row-by-row.
    """
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"fractions must sum to 1.0 (got {total}: "
            f"train={train_frac}, val={val_frac}, test={test_frac})"
        )
    fracs = (("train_frac", train_frac), ("val_frac", val_frac), ("test_frac", test_frac))
    for name, frac in fracs:
        if frac < 0 or frac > 1:
            raise ValueError(f"{name} must be in [0, 1]; got {frac}")

    rng = np.random.default_rng(seed)

    if group_column is None:
        n = len(df)
        idx = rng.permutation(n)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    if group_column not in df.columns:
        raise KeyError(f"split_fractional with group_column={group_column!r} requires that column")

    # rng.shuffle warns on non-numpy arrays (e.g. pandas StringArray); copy first.
    groups = np.array(list(df[group_column].unique()))
    rng.shuffle(groups)
    n = len(groups)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    return (
        df.loc[df[group_column].isin(train_groups)].reset_index(drop=True),
        df.loc[df[group_column].isin(val_groups)].reset_index(drop=True),
        df.loc[df[group_column].isin(test_groups)].reset_index(drop=True),
    )
