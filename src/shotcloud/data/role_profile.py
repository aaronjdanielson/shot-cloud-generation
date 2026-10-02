"""Closed-form per-player role profiles for the archetype mixture.

The role profile is a deterministic, low-dimensional summary of a
player's shot-geometry over a (causal) shot pool. It is the
canonical, interpretable input to:

* the archetype mixture network ``rho_xi(p, x_n)`` (paper §3.3),
* the defensive relevance score's player-similarity term
  ``role_sim(p', p_n)`` (paper §3.4),
* per-player diagnostics surfaced alongside the snapshot trajectory
  (paper §3.3a, "Why archetypes evolve").

This module is **stateless and learning-free**: every coordinate of
the profile is a closed-form aggregate of the input shots, so the
function output depends only on the rows passed in. Causality is the
caller's responsibility — pass the strict-past sub-frame
``shots[shots.date < t_i]``; the function makes no time decisions.

Profile composition (8-dim, ROLE_PROFILE_DIM)::

    r_p = (
        rim_rate,         # zone 0 (Restricted Area), in [0, 1]
        paint_rate,       # zone 1 (paint non-RA),    in [0, 1]
        midrange_rate,    # zone 2,                   in [0, 1]
        corner3_rate,     # zones 3+4,                in [0, 1]
        atb3_rate,        # zones 5+6+7,              in [0, 1]
        mean_dist,        # mean Euclidean distance to basket / ``DIST_SCALE``
        std_dist,         # std-dev of distance       / ``DIST_SCALE_STD``
        shot_entropy,     # Shannon entropy on coarse 8x8 grid / log(64)
    )

The first five rates partition shots and sum to 1 (unless backcourt
shots are present, which the loader already filters). The last three
are continuous unit-scale features: mean distance and its spread
characterize the player's offensive radius; the entropy distinguishes
spot-up specialists (concentrated) from movement shooters (spread).

See also :func:`shotcloud.data.snapshots.build_snapshot_store_from_shots`,
which accepts this builder via its ``role_profile_fn`` hook.
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shotcloud.data.snapshots import ROLE_PROFILE_DIM
from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

#: Names of the 8 role-profile coordinates, in the order returned by
#: :func:`build_role_profiles`.
ROLE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "rim_rate",
    "paint_rate",
    "midrange_rate",
    "corner3_rate",
    "atb3_rate",
    "mean_dist",
    "std_dist",
    "shot_entropy",
)

#: Scale factor for ``mean_dist`` (feet). ~30 ft is the typical max
#: NBA shot distance; dividing by this puts the normalized feature
#: in roughly [0, 1].
DIST_SCALE: Final[float] = 30.0

#: Scale factor for ``std_dist`` (feet). Half of ``DIST_SCALE`` since
#: per-player std is bounded above by the half-court diagonal divided
#: by 2 in practice.
DIST_SCALE_STD: Final[float] = 15.0

#: Coarse-grid resolution for the shot-entropy feature. 8x8 = 64 bins
#: gives a stable Shannon entropy from a few dozen shots upward; finer
#: grids are noisy on sparse-history players. Max possible entropy is
#: ``log(64) ~ 4.16``; the normalized feature is in [0, 1].
ENTROPY_BIN_COUNT: Final[int] = 8

# Half-court extent for the entropy histogram. Matches the model's
# default :class:`~shotcloud.grids.CourtGrid`.
_X_MIN: Final[float] = -25.0
_X_MAX: Final[float] = 25.0
_Y_MIN: Final[float] = -5.0
_Y_MAX: Final[float] = 47.0


def _shot_entropy(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Shannon entropy of (x, y) on a coarse 8x8 grid, normalized to [0, 1].

    Returns 0 for fewer than 2 shots (no spread information) so the
    feature is well-defined for sparse-history players. Empty input
    returns 0 as well.
    """
    n = x.shape[0]
    if n < 2:
        return 0.0
    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=(ENTROPY_BIN_COUNT, ENTROPY_BIN_COUNT),
        range=((_X_MIN, _X_MAX), (_Y_MIN, _Y_MAX)),
    )
    p = hist.flatten()
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nonzero = p[p > 0]
    h = float(-np.sum(nonzero * np.log(nonzero)))
    h_max = float(np.log(ENTROPY_BIN_COUNT * ENTROPY_BIN_COUNT))
    return h / h_max if h_max > 0 else 0.0


def build_role_profiles(
    shots: pd.DataFrame,
    *,
    player_col: str = "player_id",
    x_col: str = "x",
    y_col: str = "y",
    min_shots: int = 1,
    normalize: bool = True,
) -> dict[int, NDArray[np.float32]]:
    """Compute closed-form role profiles for every player in `shots`.

    Each profile is a deterministic, learning-free 8-dim summary of
    the player's shot geometry over the input shot pool. The function
    is stateless: the output for player ``p`` depends only on the
    rows of `shots` with ``player_id == p``.

    Causality is the caller's responsibility: pass
    ``shots[shots.date < t_i]`` to obtain a profile valid at anchor
    ``t_i``. The function does not inspect any date column.

    Parameters
    ----------
    shots : DataFrame
        Must have columns ``player_col``, ``x_col``, ``y_col``.
        Coordinates are expected in feet with the basket at the
        origin (the convention enforced by
        :func:`shotcloud.data.load_shots`).
    player_col, x_col, y_col : str, optional
        Column names. Defaults match :func:`load_shots` output.
    min_shots : int, default 1
        Players with fewer than this many shots are omitted from the
        result. The default (1) admits any player with at least one
        shot. Set to 50 to match the per-player KDE filter used in
        the snapshot pretraining script.
    normalize : bool, default True
        When True, distance features are divided by their unit scale
        and entropy by ``log(ENTROPY_BIN_COUNT**2)``, putting all
        coordinates in roughly [0, 1]. When False, distance features
        are reported in raw feet and entropy in raw nats.

    Returns
    -------
    dict[int, NDArray[np.float32]]
        Mapping from player ID to an ``(8,)`` float32 array. Players
        absent from `shots` (or below `min_shots`) are absent from
        the dict.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    for col in (player_col, x_col, y_col):
        if col not in shots.columns:
            raise ValueError(f"shots is missing required column {col!r}; got {list(shots.columns)}")
    if min_shots < 0:
        raise ValueError(f"min_shots must be non-negative, got {min_shots}")

    if len(shots) == 0:
        return {}

    x_all = shots[x_col].to_numpy(dtype=np.float64)
    y_all = shots[y_col].to_numpy(dtype=np.float64)

    # Cell-zone for every shot in one vectorized pass (8-zone taxonomy).
    zones = zone_from_xy_vectorized(x_all, y_all)
    distances = np.sqrt(x_all * x_all + y_all * y_all)

    # Group masks per player (avoids per-player groupby overhead).
    player_ids = shots[player_col].to_numpy()

    profiles: dict[int, NDArray[np.float32]] = {}
    unique_pids = pd.unique(player_ids)
    for pid in unique_pids:
        mask = player_ids == pid
        n = int(mask.sum())
        if n < min_shots:
            continue
        z = zones[mask]
        d = distances[mask]
        x_p = x_all[mask]
        y_p = y_all[mask]

        # Zone rates over the 8-zone taxonomy.
        # Backcourt and other invalid zones get index -1; we treat
        # them as out-of-distribution and exclude from the rate
        # denominator implicitly by counting only valid zones.
        valid = (z >= 0) & (z < N_ZONES)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        z_valid = z[valid]
        rim_rate = float((z_valid == 0).sum()) / n_valid
        paint_rate = float((z_valid == 1).sum()) / n_valid
        midrange_rate = float((z_valid == 2).sum()) / n_valid
        corner3_rate = float(((z_valid == 3) | (z_valid == 4)).sum()) / n_valid
        atb3_rate = float(((z_valid == 5) | (z_valid == 6) | (z_valid == 7)).sum()) / n_valid

        # Distance features.
        mean_dist = float(d.mean())
        std_dist = float(d.std(ddof=0)) if n > 1 else 0.0

        # Shot entropy on coarse 8x8 grid.
        entropy = _shot_entropy(x_p, y_p)

        if normalize:
            mean_dist = mean_dist / DIST_SCALE
            std_dist = std_dist / DIST_SCALE_STD
            # entropy is already normalized in _shot_entropy

        profiles[int(pid)] = np.array(
            [
                rim_rate,
                paint_rate,
                midrange_rate,
                corner3_rate,
                atb3_rate,
                mean_dist,
                std_dist,
                entropy,
            ],
            dtype=np.float32,
        )

    assert all(v.shape == (ROLE_PROFILE_DIM,) for v in profiles.values()), (
        "internal: profile shape mismatch"
    )
    return profiles


def build_role_profiles_dataframe(
    shots: pd.DataFrame,
    *,
    player_col: str = "player_id",
    x_col: str = "x",
    y_col: str = "y",
    min_shots: int = 1,
    normalize: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper returning a player-indexed DataFrame.

    Same semantics as :func:`build_role_profiles`, but the output is
    a DataFrame with one row per player and columns named by
    :data:`ROLE_FEATURE_NAMES`. Useful for diagnostics and
    interactive exploration; the dict form is what
    :func:`shotcloud.data.snapshots.build_snapshot_store_from_shots`
    consumes.
    """
    profiles = build_role_profiles(
        shots,
        player_col=player_col,
        x_col=x_col,
        y_col=y_col,
        min_shots=min_shots,
        normalize=normalize,
    )
    if not profiles:
        return pd.DataFrame(columns=[player_col, *ROLE_FEATURE_NAMES])
    rows = [
        {player_col: pid, **dict(zip(ROLE_FEATURE_NAMES, vec.tolist(), strict=True))}
        for pid, vec in profiles.items()
    ]
    df = pd.DataFrame(rows)
    return cast("pd.DataFrame", df.set_index(player_col).sort_index())


__all__ = [
    "DIST_SCALE",
    "DIST_SCALE_STD",
    "ENTROPY_BIN_COUNT",
    "ROLE_FEATURE_NAMES",
    "ROLE_PROFILE_DIM",
    "build_role_profiles",
    "build_role_profiles_dataframe",
]
