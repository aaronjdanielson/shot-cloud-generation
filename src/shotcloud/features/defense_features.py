"""Per-(opponent, snapshot) defensive feature pipeline (PR-D0.5).

Builds the team / zone / reliability blocks of the defensive query
context :math:`c_{d,n,r}` from
:mod:`docs/defense_integration_proposal.md` §4. The output tensor
mirrors the indexing of
:class:`shotcloud.models.defensive_retrieval_cache.DefensiveRetrievalCache`
so PR-D1's defensive scorer can gather features by
``(opp_idx, snapshot_idx)`` alongside the cache's allowed-shot
indices.

This is a **pure data layer** — the features are causal aggregates
of allowed-shot data, computed with recency-weighted attention.
The features themselves are not learnable; PR-D1's relevance head
will consume them as a fixed query context.

Feature layout (length :data:`DEFENSE_FEATURE_DIM` ``= 24``):

Block A — allowed-shot summary scalars (4):
    * ``allowed_mean_shot_distance``
    * ``allowed_std_shot_distance``
    * ``allowed_3pa_rate``
    * ``allowed_2pa_rate``

Block B — raw zone proportions (8):
    * ``q_rim_raw, q_paint_raw, q_mid_raw,
      q_LC3_raw, q_RC3_raw, q_LW3_raw, q_RW3_raw, q_ATB3_raw``

Block C — league-centered zone proportions (8):
    * ``q_<zone>_centered = q_d^<zone>(t) − q_bar^<zone>(t)`` where
      ``q_bar`` is the league-wide weighted mean at snapshot ``t``.

Block D — reliability (4):
    * ``log1p_allowed_count``      — sample size proxy
    * ``days_since_first_allowed`` — coverage start (0 on cold-start)
    * ``days_since_last_allowed``  — coverage freshness (0 on cold-start)
    * ``effective_sample_size``    — :math:`\\Sigma w_j` under
      exponential recency weights

Cold-start opponents (no causal allowed shots before the snapshot
anchor) get **all-zero** feature blocks; the reliability block is
the disambiguator — ``log1p_allowed_count = 0`` flags the row as
no-data and PR-D1 can route around it.

Lineup features (``v_d^lineup(n,r)``) are intentionally *not* in this
PR — they're per-shot, not per-(opp, snapshot), and the lineup-join
data path needs separate plumbing. They'll arrive in PR-D0.6 or
PR-D1.5.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized
from shotcloud.training.dataset import OpponentVocab

#: Default recency window (in days) for causal allowed-shot
#: aggregation. Same window the defensive retrieval cache uses in v1.
DEFAULT_DEFENSE_FEATURE_WINDOW_DAYS: int = 365

#: Default exponential recency half-life (in days). The downstream
#: attention scorer's half-life lives separately in PR-D1; this is
#: the half-life used only for the feature aggregates.
DEFAULT_DEFENSE_FEATURE_HALF_LIFE_DAYS: float = 90.0

# ---------------------------------------------------------------------------
# Canonical feature layout (length = 24, in this exact order)
# ---------------------------------------------------------------------------

_TEAM_SCALAR_NAMES: Final[tuple[str, ...]] = (
    "allowed_mean_shot_distance",
    "allowed_std_shot_distance",
    "allowed_3pa_rate",
    "allowed_2pa_rate",
)

#: Zone labels matching :data:`shotcloud.data.zones.ZONE_NAMES`. We
#: alias the labels here for stable feature names; the index order is
#: the same as ``ZONE_NAMES`` (0=rim, 1=paint, 2=mid, 3=LC3, 4=RC3,
#: 5=LW3, 6=RW3, 7=ATB3).
_ZONE_LABELS_FOR_FEATURES: Final[tuple[str, ...]] = (
    "rim",
    "paint",
    "mid",
    "LC3",
    "RC3",
    "LW3",
    "RW3",
    "ATB3",
)
assert len(_ZONE_LABELS_FOR_FEATURES) == N_ZONES

_ZONE_NAMES_RAW: Final[tuple[str, ...]] = tuple(f"q_{z}_raw" for z in _ZONE_LABELS_FOR_FEATURES)
_ZONE_NAMES_CENTERED: Final[tuple[str, ...]] = tuple(
    f"q_{z}_centered" for z in _ZONE_LABELS_FOR_FEATURES
)

_RELIABILITY_NAMES: Final[tuple[str, ...]] = (
    "log1p_allowed_count",
    "days_since_first_allowed",
    "days_since_last_allowed",
    "effective_sample_size",
)

#: Canonical feature-name tuple. Length must equal
#: :data:`DEFENSE_FEATURE_DIM`; changing order breaks downstream
#: consumers, so this is a stable identifier (mirrors the bucket-
#: label convention in :mod:`shotcloud.evaluation.history_buckets`).
DEFENSE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *_TEAM_SCALAR_NAMES,
    *_ZONE_NAMES_RAW,
    *_ZONE_NAMES_CENTERED,
    *_RELIABILITY_NAMES,
)

DEFENSE_FEATURE_DIM: Final[int] = len(DEFENSE_FEATURE_NAMES)

# Block slice offsets (for documentation + downstream consumers).
_TEAM_SCALAR_SLICE: Final[slice] = slice(0, len(_TEAM_SCALAR_NAMES))
_ZONE_RAW_SLICE: Final[slice] = slice(len(_TEAM_SCALAR_NAMES), len(_TEAM_SCALAR_NAMES) + N_ZONES)
_ZONE_CENTERED_SLICE: Final[slice] = slice(
    len(_TEAM_SCALAR_NAMES) + N_ZONES,
    len(_TEAM_SCALAR_NAMES) + 2 * N_ZONES,
)
_RELIABILITY_SLICE: Final[slice] = slice(
    len(_TEAM_SCALAR_NAMES) + 2 * N_ZONES,
    DEFENSE_FEATURE_DIM,
)

# Which raw-zone indices count as 3-point shots for the 3PA-rate
# convenience scalar (LC3, RC3, LW3, RW3, ATB3 = indices 3..7).
_THREE_POINT_ZONE_IDX: Final[tuple[int, ...]] = (3, 4, 5, 6, 7)


@dataclass(frozen=True)
class DefenseFeaturesConfig:
    """Configuration for the defensive feature build.

    The half-life lives here (not on the retrieval cache config)
    because it affects feature *values*, not which shots are
    retrieved. Mirrors the budget-vs-mass split that motivated PR-D0's
    decision to remove half-life from the cache hash.
    """

    shots_fingerprint: str
    anchor_dates: tuple[int, ...]
    window_days: int = DEFAULT_DEFENSE_FEATURE_WINDOW_DAYS
    half_life_days: float = DEFAULT_DEFENSE_FEATURE_HALF_LIFE_DAYS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError(f"window_days must be positive; got {self.window_days}")
        if self.half_life_days <= 0.0:
            raise ValueError(f"half_life_days must be positive; got {self.half_life_days}")

    @property
    def config_hash(self) -> str:
        """16-character hex SHA-256 of the canonical JSON payload."""
        payload = {
            "shots_fingerprint": self.shots_fingerprint,
            "anchor_dates": list(self.anchor_dates),
            "window_days": self.window_days,
            "half_life_days": self.half_life_days,
            "seed": self.seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class DefenseFeatures:
    """Per-(opponent, snapshot) defensive feature tensor + metadata.

    Attributes
    ----------
    features : Tensor of shape ``(n_opponents, n_snapshots, DEFENSE_FEATURE_DIM)``
        Float32. Cold-start (opp, snap) cells are all-zero; the
        reliability block disambiguates them from legitimately-zero
        rates.
    feature_names : tuple of str
        Length :data:`DEFENSE_FEATURE_DIM`; matches the canonical
        layout above.
    config : DefenseFeaturesConfig
        The config used to build the features. Carried for
        config-hash round-trips and downstream consistency checks.
    """

    features: Tensor
    feature_names: tuple[str, ...]
    config: DefenseFeaturesConfig

    def save(self, path: Path) -> None:
        """Atomic save: write to ``path.tmp`` then ``rename``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "features": self.features,
            "feature_names": list(self.feature_names),
            "config": asdict(self.config),
        }
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> DefenseFeatures:
        d = torch.load(path, weights_only=False)
        cfg_dict = d["config"]
        cfg_dict = {**cfg_dict, "anchor_dates": tuple(cfg_dict["anchor_dates"])}
        cfg = DefenseFeaturesConfig(**cfg_dict)
        return cls(
            features=d["features"],
            feature_names=tuple(d["feature_names"]),
            config=cfg,
        )


def _prepare_shots(
    shots_df: pd.DataFrame, opp_vocab: OpponentVocab
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter ``shots_df`` to in-vocab non-NA allowed shots and return
    ``(opp_idx, date_ord, distance, zone_idx)`` arrays sorted ascending
    by date.

    The build only needs four per-shot columns and they're all
    derivable from canonical ``load_shots()`` output:

    * ``opp_idx``     — defending team's vocab index
    * ``date_ord``    — epoch-day date (int64)
    * ``distance``    — sqrt(x² + y²) shot distance in ft
    * ``zone_idx``    — 8-zone label in ``[0, 7]`` (out-of-court → -1,
                        dropped)
    """
    if "opponent" not in shots_df.columns:
        raise ValueError(
            "shots_df must carry an 'opponent' column "
            "(populated by shotcloud.data.loaders.load_shots)"
        )
    id_to_idx = {oid: i for i, oid in enumerate(opp_vocab.ids)}
    opp_raw = shots_df["opponent"]
    opp_present = opp_raw.notna().to_numpy()
    if not opp_present.all():
        shots_df = shots_df.iloc[opp_present].reset_index(drop=True)
        opp_raw = shots_df["opponent"]
    opp_str = opp_raw.astype(str).to_numpy()
    keep = np.array([o in id_to_idx for o in opp_str], dtype=bool)
    if not keep.all():
        shots_df = shots_df.iloc[keep].reset_index(drop=True)
        opp_str = opp_str[keep]
    opp_idx = np.array([id_to_idx[o] for o in opp_str], dtype=np.int64)
    date_ord = pd.to_datetime(shots_df["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
    x = shots_df["x"].to_numpy(dtype=np.float64)
    y = shots_df["y"].to_numpy(dtype=np.float64)
    distance = np.sqrt(x * x + y * y)
    zone_idx = zone_from_xy_vectorized(x, y)
    # Drop out-of-court shots (zone == -1) from the aggregation —
    # they're noise for defensive feasibility.
    in_court = zone_idx >= 0
    if not in_court.all():
        opp_idx = opp_idx[in_court]
        date_ord = date_ord[in_court]
        distance = distance[in_court]
        zone_idx = zone_idx[in_court]
    order = np.argsort(date_ord, kind="stable")
    return opp_idx[order], date_ord[order], distance[order], zone_idx[order]


def _league_zone_baseline(
    *,
    date_ord: np.ndarray,  # (N,) sorted by date
    zone_idx: np.ndarray,  # (N,)
    weights: np.ndarray,  # (N,) recency weights for this snapshot
    i_hi: int,  # global slice [:i_hi] is the causal pool
    i_lo_win: int,  # global slice [i_lo_win:i_hi] is the recency window
) -> np.ndarray:
    """League-wide weighted zone proportions at one snapshot.

    Returns an ``(N_ZONES,)`` float64 array summing to 1 (or zeros if
    the window has no shots at all). Used as the centering baseline
    for each opponent's zone rates.
    """
    del date_ord  # unused; kept for symmetry with the call site
    if i_lo_win >= i_hi:
        return np.zeros(N_ZONES, dtype=np.float64)
    win_zone = zone_idx[i_lo_win:i_hi]
    win_w = weights[i_lo_win:i_hi]
    totals = np.zeros(N_ZONES, dtype=np.float64)
    np.add.at(totals, win_zone, win_w)
    total_w = float(totals.sum())
    if total_w <= 0.0:
        return np.zeros(N_ZONES, dtype=np.float64)
    # cast(): NumPy 2.x stubs leak Any through arithmetic chains;
    # explicit cast preserves the declared return type. (Same pattern
    # documented in CLAUDE.md "Known sharp edges".)
    from typing import cast

    return cast(np.ndarray, totals / total_w)


def _build_one_snapshot(
    *,
    s_idx: int,
    anchor: int,
    opp_idx: np.ndarray,
    date_ord: np.ndarray,
    distance: np.ndarray,
    zone_idx: np.ndarray,
    n_opps: int,
    config: DefenseFeaturesConfig,
    out: np.ndarray,  # (n_opps, n_snaps, DEFENSE_FEATURE_DIM)
) -> None:
    """Fill ``out[:, s_idx, :]`` with each opp's feature vector at
    snapshot ``anchor``.

    The function mutates ``out`` rather than returning; this keeps
    memory pressure flat (we allocate one big tensor once and write
    into it).
    """
    # Causal upper bound: date < anchor.
    i_hi = int(np.searchsorted(date_ord, anchor, side="left"))
    if i_hi == 0:
        # Pre-data snapshot — every opp is cold-start at this anchor.
        return
    # Recency lower bound: date >= anchor − window.
    i_lo_win = int(np.searchsorted(date_ord, anchor - config.window_days, side="left"))
    if i_lo_win >= i_hi:
        return
    win_opp = opp_idx[i_lo_win:i_hi]
    win_date = date_ord[i_lo_win:i_hi]
    win_dist = distance[i_lo_win:i_hi]
    win_zone = zone_idx[i_lo_win:i_hi]
    # Exponential recency weights.
    delta = (anchor - win_date).astype(np.float64)
    log2_per_day = float(np.log(2.0)) / config.half_life_days
    win_w = np.exp(-delta * log2_per_day)
    # League baseline for centering — single computation per snapshot.
    league_q = _league_zone_baseline(
        date_ord=date_ord,
        zone_idx=zone_idx,
        weights=np.concatenate([np.zeros(i_lo_win, dtype=np.float64), win_w]),
        i_hi=i_hi,
        i_lo_win=i_lo_win,
    )
    # Per-opponent aggregation.
    for d_idx in range(n_opps):
        sel = win_opp == d_idx
        if not sel.any():
            # Cold-start opp at this snapshot — leave as all zeros
            # (already initialized).
            continue
        d_w = win_w[sel]
        d_dist = win_dist[sel]
        d_zone = win_zone[sel]
        d_date = win_date[sel]
        total_w = float(d_w.sum())
        if total_w <= 0.0:
            continue
        # Block A — allowed-shot summary scalars.
        mean_dist = float(np.sum(d_w * d_dist) / total_w)
        # Weighted std: sqrt(E[X²] - E[X]²), clamped at 0 for numerical safety.
        mean_sq = float(np.sum(d_w * d_dist * d_dist) / total_w)
        var = max(0.0, mean_sq - mean_dist * mean_dist)
        std_dist = float(np.sqrt(var))
        # Block B — raw zone proportions.
        zone_totals = np.zeros(N_ZONES, dtype=np.float64)
        np.add.at(zone_totals, d_zone, d_w)
        q_raw = zone_totals / total_w
        three_rate = float(q_raw[list(_THREE_POINT_ZONE_IDX)].sum())
        two_rate = float(1.0 - three_rate)
        # Block C — centered zone proportions.
        q_centered = q_raw - league_q
        # Block D — reliability.
        log1p_count = float(np.log1p(int(sel.sum())))
        days_since_first = float(anchor - int(d_date.min()))
        days_since_last = float(anchor - int(d_date.max()))
        ess = total_w  # Σ w_j (a simple effective-sample-size proxy)
        # Write into the output tensor.
        feat = out[d_idx, s_idx]
        feat[_TEAM_SCALAR_SLICE] = [mean_dist, std_dist, three_rate, two_rate]
        feat[_ZONE_RAW_SLICE] = q_raw
        feat[_ZONE_CENTERED_SLICE] = q_centered
        feat[_RELIABILITY_SLICE] = [
            log1p_count,
            days_since_first,
            days_since_last,
            ess,
        ]


def build_defense_features(
    *,
    shots_df: pd.DataFrame,
    opp_vocab: OpponentVocab,
    anchor_dates: np.ndarray,
    config: DefenseFeaturesConfig,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> DefenseFeatures:
    """Build (or disk-load) per-(opponent, snapshot) defensive features.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Must carry ``x, y, opponent, date`` columns (canonical
        ``load_shots()`` output). Rows with missing opponent, out-of-
        vocab opponent, or out-of-court coordinates are dropped.
    opp_vocab : OpponentVocab
        Defending-team vocabulary; defines the row indexing.
    anchor_dates : np.ndarray of shape ``(S,)``
        Snapshot anchor dates in epoch days.
    config : DefenseFeaturesConfig
    cache_dir : Path or None
        If given, the feature tensor is loaded from
        ``cache_dir/defense_features_{config.config_hash}.pt`` when
        that file exists (and ``rebuild=False``); otherwise written
        atomically after a fresh build.
    rebuild : bool
        Force a rebuild even when a cached file exists.

    Returns
    -------
    DefenseFeatures
    """
    if anchor_dates.ndim != 1:
        raise ValueError(f"anchor_dates must be 1D; got shape {anchor_dates.shape}")
    if anchor_dates.dtype != np.int64:
        anchor_dates = anchor_dates.astype(np.int64)
    if tuple(config.anchor_dates) != tuple(int(d) for d in anchor_dates):
        raise ValueError(
            "config.anchor_dates must equal the anchor_dates argument "
            "(the hash depends on them; mismatch would corrupt the cache key)"
        )

    if cache_dir is not None and not rebuild:
        path = cache_dir / f"defense_features_{config.config_hash}.pt"
        if path.exists():
            return DefenseFeatures.load(path)

    n_opps = len(opp_vocab)
    n_snaps = anchor_dates.shape[0]
    out = np.zeros((n_opps, n_snaps, DEFENSE_FEATURE_DIM), dtype=np.float32)

    opp_idx, date_ord, distance, zone_idx = _prepare_shots(shots_df, opp_vocab)

    for s_idx, anchor in enumerate(anchor_dates.tolist()):
        _build_one_snapshot(
            s_idx=s_idx,
            anchor=int(anchor),
            opp_idx=opp_idx,
            date_ord=date_ord,
            distance=distance,
            zone_idx=zone_idx,
            n_opps=n_opps,
            config=config,
            out=out,
        )

    features = DefenseFeatures(
        features=torch.from_numpy(out),
        feature_names=DEFENSE_FEATURE_NAMES,
        config=config,
    )
    if cache_dir is not None:
        path = cache_dir / f"defense_features_{config.config_hash}.pt"
        features.save(path)
    return features


__all__ = [
    "DEFAULT_DEFENSE_FEATURE_HALF_LIFE_DAYS",
    "DEFAULT_DEFENSE_FEATURE_WINDOW_DAYS",
    "DEFENSE_FEATURE_DIM",
    "DEFENSE_FEATURE_NAMES",
    "DefenseFeatures",
    "DefenseFeaturesConfig",
    "build_defense_features",
]
