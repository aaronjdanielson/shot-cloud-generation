"""Per-(group, snapshot, opponent) matchup Δ̂ feature pipeline (Tier 2a-v2).

Builds the residualized similar-player defensive response feature

.. math::

    \\Delta^{\\mathrm{int}}_{k,d,z}(t)
    = P^{\\mathrm{vs}}_{k,d}(z,t)
    - P^{\\mathrm{base}}_{k}(z,t)
    - P^{\\mathrm{allow}}_{d}(z,t)
    + P_{\\mathrm{lg}}(z,t)

shrunk by

.. math::

    \\widehat\\Delta_{k,d,z}(t) = \\kappa_{k,d}(t) \\Delta^{\\mathrm{int}}_{k,d,z}(t),
    \\qquad
    \\kappa_{k,d}(t) = \\frac{N^{\\mathrm{eff}}_{k,d}(t)}{N^{\\mathrm{eff}}_{k,d}(t) + \\tau}.

The "players like p" at snapshot ``t`` are now defined as
**discrete K-means groups** over the trait vector
(:class:`shotcloud.data.player_traits.PlayerTraitsTable`), not the
per-(p, t) top-L cosine peer set the v1 build used. Group ``k`` at
snapshot ``t`` aggregates ~n_players/K members; for K=5 and ~1500
players, each (group, opp, zone) cell sees ~5–10× more shots than the
v1 per-player setup, which was too noisy for the third-order
interaction the residualization isolates (see the 2026-06-03 log
entry).

Aggregation (group-level):

* ``P^vs_{k,d}(z)`` — recency-weighted zone histogram of shots taken
  by **all players in group k** against opponent ``d`` in the causal
  window ``[t - window_days, t)``.
* ``P^base_k(z)``   — same group, *any* opponent.
* ``P^allow_d(z)``  — *any* player, opponent ``d``.
* ``P^lg(z)``       — league-wide.
* ``N^eff_{k,d}(t)`` — sum of recency weights for group-vs-``d`` shots.
  Cells with ``N^eff = 0`` (no causal group-vs-opponent shots) get
  ``Δ̂ = 0``; downstream this collapses the matchup term to a no-op,
  the cold-start-safe fallback.

Limit cases:

* ``K = 1`` (one group, all players): ``P^vs_{1,d}(z) = P^allow_d(z)``
  exactly, so ``Δ^int = (P^allow − P^lg) − (P^allow − P^lg) = 0``
  identically. The matchup term collapses to a no-op — this is the
  orthogonal-decomposition identity made explicit.
* ``K`` large: groups shrink toward per-player; signal becomes
  identification-limited (the v1 per-player setup empirically).
* The sweet spot is moderate K (default 5) where each group is large
  enough for low-variance Δ̂ but small enough to expose real
  player-type-conditional structure.

The output ``delta_hat`` is broadcast to per-player shape
``(n_players, n_snapshots, n_opps, N_ZONES)`` by gathering
``delta_hat_group[group(p, t), t, d, :]`` — players in the same group
share the same Δ̂ row by construction. The trainer's per-row gather
in :class:`MatchupReweightingDefense` is unchanged.

Cache-key parity is the load-bearing invariant: the config hash
covers the shots fingerprint, the opponent vocabulary hash, the
anchor-date tuple, ``grouping_K``, ``grouping_seed``, and the
``(window_days, half_life_days, tau, seed)`` quad. Changing any of
these changes the hash; the trainer rebuilds on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch import Tensor

from shotcloud.data.player_traits import PlayerTraitsTable
from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized
from shotcloud.training.dataset import OpponentVocab, PlayerVocab

#: Default recency window (in days) for causal peer-shot aggregation.
#: Matches the D-lite defense feature window so the two defense
#: channels are temporally aligned for the A vs B vs C ablation.
DEFAULT_MATCHUP_WINDOW_DAYS: int = 365

#: Default exponential recency half-life (in days). Matches the
#: D-lite half-life.
DEFAULT_MATCHUP_HALF_LIFE_DAYS: float = 90.0

#: Default shrinkage scale τ for κ = N_eff / (N_eff + τ). At τ=20,
#: a group-vs-opponent cell needs ~20 recency-weighted shots before
#: its Δ̂ reaches half of Δ^int. With K=5 groups and ~300 players
#: per group, N^eff per (group, opp) cell is typically in the
#: low-hundreds, so the default τ leaves κ in [0.85, 0.95] — light
#: shrinkage on well-populated cells, heavy on cold cells.
DEFAULT_MATCHUP_TAU: float = 20.0

#: Default number of K-means groups over the trait vector. At K=5
#: and ~1500 players the average group is ~300 players → ~5–10× more
#: shots per (group, opp, zone) cell than the v1 per-player setup
#: that empirically underperformed D-lite-zone.
DEFAULT_MATCHUP_GROUPING_K: int = 5

#: Default seed for the K-means clusterer. Deterministic given seed +
#: traits + K; carried in the config hash so cache invalidates on
#: change.
DEFAULT_MATCHUP_GROUPING_SEED: int = 0


@dataclass(frozen=True)
class MatchupFeaturesConfig:
    """Configuration for the (group-level) matchup-features build.

    The hash covers everything that can change the feature values:
    the shots fingerprint, the opponent vocabulary hash, the
    anchor-date tuple, the trait-table hash (so re-clustering on
    different traits invalidates), the K-means K + seed, and the
    (window, half_life, tau, seed) quad. A change to any of these
    invalidates the cached artifact.
    """

    shots_fingerprint: str
    traits_hash: str
    opp_vocab_hash: str
    anchor_dates: tuple[int, ...]
    # ``grouping_K``'s mixedCase preserved to match the ``--matchup-K`` CLI flag.
    grouping_K: int = DEFAULT_MATCHUP_GROUPING_K  # noqa: N815
    grouping_seed: int = DEFAULT_MATCHUP_GROUPING_SEED
    window_days: int = DEFAULT_MATCHUP_WINDOW_DAYS
    half_life_days: float = DEFAULT_MATCHUP_HALF_LIFE_DAYS
    tau: float = DEFAULT_MATCHUP_TAU
    seed: int = 0

    def __post_init__(self) -> None:
        if self.grouping_K < 1:
            raise ValueError(f"grouping_K must be ≥ 1; got {self.grouping_K}")
        if self.window_days <= 0:
            raise ValueError(f"window_days must be positive; got {self.window_days}")
        if self.half_life_days <= 0.0:
            raise ValueError(f"half_life_days must be positive; got {self.half_life_days}")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive; got {self.tau}")

    @property
    def config_hash(self) -> str:
        """16-character hex SHA-256 of the canonical JSON payload."""
        payload = {
            "shots_fingerprint": self.shots_fingerprint,
            "traits_hash": self.traits_hash,
            "opp_vocab_hash": self.opp_vocab_hash,
            "anchor_dates": list(self.anchor_dates),
            "grouping_K": self.grouping_K,
            "grouping_seed": self.grouping_seed,
            "window_days": self.window_days,
            "half_life_days": self.half_life_days,
            "tau": self.tau,
            "seed": self.seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def traits_table_hash(traits: PlayerTraitsTable) -> str:
    """Stable 16-char SHA-256 hash of the trait tensor's bytes.

    Used by :class:`MatchupFeaturesConfig` to invalidate the cache
    when the underlying trait values change (e.g. snapshot store
    rebuilt, game logs extended, bio table updated).
    """
    arr = np.ascontiguousarray(traits.traits, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def assign_player_groups(
    *,
    traits: PlayerTraitsTable,
    K: int,
    seed: int = DEFAULT_MATCHUP_GROUPING_SEED,
) -> NDArray[np.int64]:
    """K-means cluster the trait vector at each snapshot into K groups.

    Returns a ``(n_players, n_snapshots)`` int64 array of group
    assignments in ``[0, K)``. Per-snapshot clustering is causal-by-
    construction (each column uses only that snapshot's trait values,
    which are themselves built from data with date < anchor in
    :func:`build_player_traits_table`).

    The trait vector is already z-score-normalized inside
    :class:`PlayerTraitsTable`, so KMeans operates on standardized
    features directly.

    Parameters
    ----------
    traits : PlayerTraitsTable
        Causal trait tensor; shape ``(n_players, n_snapshots, trait_dim)``.
    K : int
        Number of groups. ``K=1`` is the league-marginal limit
        (matchup term collapses to zero); ``K`` ≥ ``n_players`` is
        rejected (each player would be its own group, defeating the
        purpose).
    seed : int
        K-means random_state. Determines initialization; for K-means++
        with a fixed seed the assignment is deterministic.

    Returns
    -------
    NDArray[np.int64] of shape ``(n_players, n_snapshots)``
    """
    if K < 1:
        raise ValueError(f"K must be ≥ 1; got {K}")
    n_players = traits.n_players
    n_snapshots = traits.n_snapshots
    if n_players < K:
        raise ValueError(
            f"K={K} exceeds n_players={n_players}; cannot form more groups than players"
        )

    groups = np.zeros((n_players, n_snapshots), dtype=np.int64)
    if K == 1:
        # All players in one group at every snapshot. With the
        # group-level aggregation, P^vs = P^allow exactly → Δ^int = 0.
        # The output stays at all-zeros; matchup term is a no-op.
        return groups

    for m in range(n_snapshots):
        X = traits.traits[:, m, :].astype(np.float64)
        # NaN-safe: any rows with NaN are zero-filled (matches the
        # imputation convention upstream in build_player_traits_table).
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # n_init=10 to mitigate K-means local-minima sensitivity; with
        # a fixed seed each init is reproducible.
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        groups[:, m] = km.fit_predict(X).astype(np.int64)
    return groups


@dataclass
class MatchupFeatures:
    """Per-(player, snapshot, opponent) matchup Δ̂_{p,d,z}(t) tensor.

    Attributes
    ----------
    delta_hat : Tensor of shape ``(n_players, n_snapshots, n_opps, N_ZONES)``
        Float32. Shrunk residualized matchup effect. Cells with no
        causal peer-vs-opponent evidence (``N^eff = 0``) are exactly
        zero, so the downstream ``β_match · Δ̂`` term is a no-op for
        those rows — cold-start-safe by construction.
    delta_int : Tensor of shape ``(n_players, n_snapshots, n_opps, N_ZONES)``
        Float32. Pre-shrink Δ^int. Carried alongside Δ̂ for
        diagnostics (lets us see how much shrinkage is doing on the
        Δ̂ = κ · Δ^int factorization).
    n_eff : Tensor of shape ``(n_players, n_snapshots, n_opps)``
        Float32. Sum of recency weights for peer-vs-opponent shots
        in the causal window. Used both for the shrinkage κ and for
        ESS-bucket falsification (is the matchup signal concentrated
        where evidence is strong?).
    config : MatchupFeaturesConfig
        The config used to build the features. Carried for hash
        round-trips and downstream consistency checks.
    """

    delta_hat: Tensor
    delta_int: Tensor
    n_eff: Tensor
    config: MatchupFeaturesConfig

    def save(self, path: Path) -> None:
        """Atomic save: write to ``path.tmp`` then ``rename``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "delta_hat": self.delta_hat,
            "delta_int": self.delta_int,
            "n_eff": self.n_eff,
            "config": asdict(self.config),
        }
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> MatchupFeatures:
        d = torch.load(path, weights_only=False)
        cfg_dict = d["config"]
        cfg_dict = {**cfg_dict, "anchor_dates": tuple(cfg_dict["anchor_dates"])}
        cfg = MatchupFeaturesConfig(**cfg_dict)
        return cls(
            delta_hat=d["delta_hat"],
            delta_int=d["delta_int"],
            n_eff=d["n_eff"],
            config=cfg,
        )


def _prepare_shots(
    shots_df: pd.DataFrame,
    player_vocab: PlayerVocab,
    opp_vocab: OpponentVocab,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Filter ``shots_df`` to in-vocab non-NA shots and return
    ``(player_idx, opp_idx, date_ord, zone_idx)`` arrays sorted
    ascending by date.

    Only shots whose player AND opponent are both in the respective
    vocabs survive; rows with out-of-court coordinates (zone = -1) are
    dropped because the matchup aggregation is zone-conditional.
    """
    if "opponent" not in shots_df.columns:
        raise ValueError(
            "shots_df must carry an 'opponent' column "
            "(populated by shotcloud.data.loaders.load_shots)"
        )
    if "player_id" not in shots_df.columns:
        raise ValueError("shots_df must carry a 'player_id' column")

    player_id_to_idx = {pid: i for i, pid in enumerate(player_vocab.ids)}
    opp_id_to_idx = {oid: i for i, oid in enumerate(opp_vocab.ids)}

    # Player filter — keep rows whose player_id is in the player vocab.
    pid_raw = shots_df["player_id"]
    pid_present = pid_raw.notna().to_numpy()
    if not pid_present.all():
        shots_df = shots_df.iloc[pid_present].reset_index(drop=True)
    pid_str = shots_df["player_id"].astype(str).to_numpy()
    keep_p = np.array([p in player_id_to_idx for p in pid_str], dtype=bool)
    if not keep_p.all():
        shots_df = shots_df.iloc[keep_p].reset_index(drop=True)
        pid_str = pid_str[keep_p]

    # Opponent filter.
    opp_raw = shots_df["opponent"]
    opp_present = opp_raw.notna().to_numpy()
    if not opp_present.all():
        shots_df = shots_df.iloc[opp_present].reset_index(drop=True)
        pid_str = pid_str[opp_present]
    opp_str = shots_df["opponent"].astype(str).to_numpy()
    keep_o = np.array([o in opp_id_to_idx for o in opp_str], dtype=bool)
    if not keep_o.all():
        shots_df = shots_df.iloc[keep_o].reset_index(drop=True)
        pid_str = pid_str[keep_o]
        opp_str = opp_str[keep_o]

    player_idx = np.array([player_id_to_idx[p] for p in pid_str], dtype=np.int64)
    opp_idx = np.array([opp_id_to_idx[o] for o in opp_str], dtype=np.int64)
    date_ord = pd.to_datetime(shots_df["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
    x = shots_df["x"].to_numpy(dtype=np.float64)
    y = shots_df["y"].to_numpy(dtype=np.float64)
    zone_idx = zone_from_xy_vectorized(x, y).astype(np.int64)

    in_court = zone_idx >= 0
    if not in_court.all():
        player_idx = player_idx[in_court]
        opp_idx = opp_idx[in_court]
        date_ord = date_ord[in_court]
        zone_idx = zone_idx[in_court]

    order = np.argsort(date_ord, kind="stable")
    return player_idx[order], opp_idx[order], date_ord[order], zone_idx[order]


def _build_one_snapshot(
    *,
    s_idx: int,
    anchor: int,
    player_idx: NDArray[np.int64],
    opp_idx: NDArray[np.int64],
    date_ord: NDArray[np.int64],
    zone_idx: NDArray[np.int64],
    groups_at_snap: NDArray[np.int64],  # (n_players,) — k-means assignments
    K: int,
    n_players: int,
    n_opps: int,
    config: MatchupFeaturesConfig,
    delta_int_out: NDArray[np.float32],  # (n_players, n_snaps, n_opps, N_ZONES)
    delta_hat_out: NDArray[np.float32],
    n_eff_out: NDArray[np.float32],  # (n_players, n_snaps, n_opps)
) -> None:
    """Fill the ``s_idx`` slice of the output tensors with Δ̂, Δ^int,
    and N^eff for every (player, opponent) pair at this snapshot.

    Players in the same group share their (group, opp, zone) Δ̂
    row by construction; the per-player output tensor broadcasts
    ``Δ̂_group[group(p), d, z]`` back across each member.

    Empty-window snapshots leave the slice at zeros (initialized
    upstream).
    """
    i_hi = int(np.searchsorted(date_ord, anchor, side="left"))
    if i_hi == 0:
        return
    i_lo = int(np.searchsorted(date_ord, anchor - config.window_days, side="left"))
    if i_lo >= i_hi:
        return

    win_player = player_idx[i_lo:i_hi]
    win_opp = opp_idx[i_lo:i_hi]
    win_zone = zone_idx[i_lo:i_hi]
    win_date = date_ord[i_lo:i_hi]

    log2_per_day = float(np.log(2.0)) / config.half_life_days
    w = np.exp(-((anchor - win_date).astype(np.float64)) * log2_per_day)

    # Aggregate shot weights directly at the (group, opp, zone) level —
    # no intermediate (n_players, n_opps, n_zones) tensor needed. For
    # each in-window shot, the contributing group is groups_at_snap of
    # its player.
    win_group = groups_at_snap[win_player]  # (n_win,) in [0, K)
    T_group_flat = np.zeros(K * n_opps * N_ZONES, dtype=np.float64)
    flat_idx = (win_group * n_opps + win_opp) * N_ZONES + win_zone
    np.add.at(T_group_flat, flat_idx, w)
    T_group = T_group_flat.reshape(K, n_opps, N_ZONES)  # (K, n_opps, N_ZONES)

    # Group marginals.
    T_group_d = T_group.sum(axis=2)  # (K, n_opps)  — group-vs-opp totals
    T_group_z = T_group.sum(axis=1)  # (K, N_ZONES) — group base zone totals
    T_group_total = T_group_z.sum(axis=1)  # (K,) — group totals
    # League marginals (sum over groups).
    T_dz = T_group.sum(axis=0)  # (n_opps, N_ZONES)
    T_d = T_dz.sum(axis=1)  # (n_opps,)
    T_z = T_group_z.sum(axis=0)  # (N_ZONES,)
    T_total = float(T_z.sum())

    # Cell probabilities. Safe divide: any all-zero cell gets P = 0,
    # and we then zero out Δ̂ for the matching N_eff = 0 cells.
    eps = 1e-12
    P_vs = T_group / np.maximum(T_group_d[:, :, None], eps)  # (K, n_opps, N_ZONES)
    P_base = T_group_z / np.maximum(T_group_total[:, None], eps)  # (K, N_ZONES)
    if T_total > 0.0:
        P_allow = T_dz / np.maximum(T_d[:, None], eps)  # (n_opps, N_ZONES)
        P_lg = T_z / T_total  # (N_ZONES,)
    else:
        P_allow = np.zeros_like(T_dz)
        P_lg = np.zeros(N_ZONES, dtype=np.float64)

    # Δ^int and shrinkage at group level.
    delta_int_group = (
        P_vs - P_base[:, None, :] - P_allow[None, :, :] + P_lg[None, None, :]
    )  # (K, n_opps, N_ZONES)
    kappa_group = T_group_d / (T_group_d + config.tau)  # (K, n_opps)
    delta_hat_group = kappa_group[:, :, None] * delta_int_group  # (K, n_opps, N_ZONES)

    # Zero out empty-evidence cells at the group level.
    zero_mask_group = T_group_d == 0.0
    delta_int_group[zero_mask_group] = 0.0
    delta_hat_group[zero_mask_group] = 0.0

    # Broadcast group-level Δ̂ back to per-player by gathering
    # ``Δ̂_group[group(p), :, :]`` for each player p.
    delta_int_s = delta_int_group[groups_at_snap]  # (n_players, n_opps, N_ZONES)
    delta_hat_s = delta_hat_group[groups_at_snap]  # (n_players, n_opps, N_ZONES)
    n_eff_s = T_group_d[groups_at_snap]  # (n_players, n_opps)

    delta_int_out[:, s_idx, :, :] = delta_int_s.astype(np.float32)
    delta_hat_out[:, s_idx, :, :] = delta_hat_s.astype(np.float32)
    n_eff_out[:, s_idx, :] = n_eff_s.astype(np.float32)


def build_matchup_features(
    *,
    shots_df: pd.DataFrame,
    groups: NDArray[np.int64],
    opp_vocab: OpponentVocab,
    player_vocab: PlayerVocab,
    anchor_dates: NDArray[np.int64],
    config: MatchupFeaturesConfig,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> MatchupFeatures:
    """Build (or disk-load) per-(group, snapshot, opponent) matchup
    Δ̂_{k,d,z}(t) features, broadcast back to per-player shape.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Canonical ``load_shots`` output with at least
        ``player_id, opponent, x, y, date`` columns. Rows with missing
        player/opponent, out-of-vocab player/opponent, or out-of-court
        coordinates are dropped.
    groups : np.ndarray of shape ``(n_players, n_snapshots)``
        K-means group assignments produced by
        :func:`assign_player_groups`. Indexed positionally into
        ``player_vocab.ids``; values are in ``[0, config.grouping_K)``.
    opp_vocab : OpponentVocab
    player_vocab : PlayerVocab
    anchor_dates : np.ndarray of shape ``(S,)``
        Snapshot anchor dates in epoch days. Must equal
        ``config.anchor_dates`` (used in the hash).
    config : MatchupFeaturesConfig
    cache_dir : Path or None
        If given, the feature artifact is loaded from
        ``cache_dir/matchup_features_{config.config_hash}.pt`` when
        present (and ``rebuild=False``); otherwise written atomically
        after a fresh build.
    rebuild : bool
        Force a rebuild even when a cached file exists.

    Returns
    -------
    MatchupFeatures
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
    n_players = len(player_vocab)
    if groups.shape != (n_players, anchor_dates.shape[0]):
        raise ValueError(
            f"groups must have shape (n_players={n_players}, "
            f"n_snapshots={anchor_dates.shape[0]}); got {tuple(groups.shape)}"
        )
    K = config.grouping_K
    if groups.size > 0:
        max_g = int(groups.max())
        if max_g >= K:
            raise ValueError(
                f"groups contains value {max_g} but config.grouping_K={K}; "
                "config and group assignments are out of sync"
            )
        if int(groups.min()) < 0:
            raise ValueError("groups must be non-negative")

    if cache_dir is not None and not rebuild:
        path = cache_dir / f"matchup_features_{config.config_hash}.pt"
        if path.exists():
            return MatchupFeatures.load(path)

    n_opps = len(opp_vocab)
    n_snaps = int(anchor_dates.shape[0])

    delta_hat = np.zeros((n_players, n_snaps, n_opps, N_ZONES), dtype=np.float32)
    delta_int = np.zeros((n_players, n_snaps, n_opps, N_ZONES), dtype=np.float32)
    n_eff = np.zeros((n_players, n_snaps, n_opps), dtype=np.float32)

    player_idx, opp_idx, date_ord, zone_idx = _prepare_shots(shots_df, player_vocab, opp_vocab)

    for s_idx, anchor in enumerate(anchor_dates.tolist()):
        _build_one_snapshot(
            s_idx=s_idx,
            anchor=int(anchor),
            player_idx=player_idx,
            opp_idx=opp_idx,
            date_ord=date_ord,
            zone_idx=zone_idx,
            groups_at_snap=groups[:, s_idx],
            K=K,
            n_players=n_players,
            n_opps=n_opps,
            config=config,
            delta_int_out=delta_int,
            delta_hat_out=delta_hat,
            n_eff_out=n_eff,
        )

    features = MatchupFeatures(
        delta_hat=torch.from_numpy(delta_hat),
        delta_int=torch.from_numpy(delta_int),
        n_eff=torch.from_numpy(n_eff),
        config=config,
    )
    if cache_dir is not None:
        path = cache_dir / f"matchup_features_{config.config_hash}.pt"
        features.save(path)
    return features


__all__ = [
    "DEFAULT_MATCHUP_GROUPING_K",
    "DEFAULT_MATCHUP_GROUPING_SEED",
    "DEFAULT_MATCHUP_HALF_LIFE_DAYS",
    "DEFAULT_MATCHUP_TAU",
    "DEFAULT_MATCHUP_WINDOW_DAYS",
    "MatchupFeatures",
    "MatchupFeaturesConfig",
    "assign_player_groups",
    "build_matchup_features",
    "traits_table_hash",
]
