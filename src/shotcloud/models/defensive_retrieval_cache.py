"""Per-(opponent, snapshot) causal pool of shots allowed by each defense.

:class:`DefensiveRetrievalCache` supplies the allowed-shot pool
:math:`\\mathcal D_d^{<t_n}` of
:class:`~shotcloud.models.continuous_adaptive_defensive.ContinuousAdaptiveDefensiveField`,
a kernel-based opponent reweighting field evaluated as an alternative to
zone-level reweighting. For opponent :math:`d` and snapshot anchor
:math:`t_n`, the pool holds the shots attempted *against* :math:`d` in
the recency window ``[t_n − defensive_recency_window_days, t_n)``, most
recent first, up to ``defensive_support_max``. Every shot in the pool
predates the anchor, so the pool is causal by construction.

This module is a pure data layer. Unlike the offensive
:mod:`shotcloud.models.retrieval_cache`, it has **no pooled
component**: defense is modelled as opponent-specific feasibility, not
as borrowing across opponents. Opponents with no causal allowed shots
get an empty mask, and the defensive field then contributes zero.

Determinism notes (mirroring the offensive cache):

* Within-window ranking uses ``np.argsort(-date, kind="stable")``,
  not ``argpartition``, so ties resolve to the lower original index
  reproducibly across NumPy versions.
* The cache file name embeds a 16-char config hash that changes
  whenever any retrieval-defining field changes, so a stale cache
  cannot be silently reused under a different config.
* The cache config carries *only* fields that affect cache contents:
  ``shots_fingerprint, anchor_dates, defensive_support_max,
  defensive_recency_window_days, ranking_kind, seed``. Attention
  hyperparameters of the defensive field (such as its recency
  half-life) belong to the field, so changing them does not invalidate
  cache files whose indices are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from shotcloud.training.dataset import OpponentVocab

#: Default per-(opponent, snapshot) allowed-shot cap.
DEFAULT_DEFENSIVE_SUPPORT_MAX: int = 1000

#: Default recency window in days. Longer than the offensive pooled
#: window because opponent schemes turn over more slowly than player
#: shot diets.
DEFAULT_DEFENSIVE_RECENCY_WINDOW_DAYS: int = 365

#: Ranking used within the recency window; ``"recency"`` (most recent
#: first) is the only implemented option.
DEFAULT_DEFENSIVE_RANKING: str = "recency"


@dataclass(frozen=True)
class DefensiveRetrievalCacheConfig:
    """All defensive retrieval-defining settings, frozen and hashed.

    Two caches with the same ``config_hash`` are guaranteed to have
    identical contents *given the same shots DataFrame*. The caller is
    responsible for keeping ``shots_fingerprint`` in sync with the
    actual data file.
    """

    shots_fingerprint: str
    anchor_dates: tuple[int, ...]
    defensive_support_max: int = DEFAULT_DEFENSIVE_SUPPORT_MAX
    defensive_recency_window_days: int = DEFAULT_DEFENSIVE_RECENCY_WINDOW_DAYS
    ranking_kind: str = DEFAULT_DEFENSIVE_RANKING
    seed: int = 0

    def __post_init__(self) -> None:
        if self.defensive_support_max <= 0:
            raise ValueError(
                f"defensive_support_max must be positive; got {self.defensive_support_max}"
            )
        if self.defensive_recency_window_days <= 0:
            raise ValueError(
                "defensive_recency_window_days must be positive; got "
                f"{self.defensive_recency_window_days}"
            )
        if self.ranking_kind != DEFAULT_DEFENSIVE_RANKING:
            raise ValueError(
                f"unknown ranking_kind={self.ranking_kind!r}; only "
                f"{DEFAULT_DEFENSIVE_RANKING!r} is implemented"
            )

    @property
    def config_hash(self) -> str:
        """16-character hex SHA-256 of the canonical JSON payload."""
        payload = {
            "shots_fingerprint": self.shots_fingerprint,
            "anchor_dates": list(self.anchor_dates),
            "defensive_support_max": self.defensive_support_max,
            "defensive_recency_window_days": self.defensive_recency_window_days,
            "ranking_kind": self.ranking_kind,
            "seed": self.seed,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class DefensiveRetrievalCache:
    """Precomputed defensive retrieval indices and the allowed-shot table they reference."""

    config: DefensiveRetrievalCacheConfig
    #: ``(N_global, 2)`` float32 — coordinates of every allowed shot
    #: in the opponent vocab, sorted ascending by ``global_dates``.
    global_xy: Tensor
    #: ``(N_global,)`` int64 — epoch-day date of each allowed shot.
    global_dates: Tensor
    #: ``(N_global,)`` int64 — opponent vocab index of the *defending*
    #: team for each allowed shot. (The defensive cache's analog of
    #: ``global_shooter_idx`` in the offensive cache.)
    global_opponent_idx: Tensor
    #: ``(n_opponents, n_snapshots, M_def)`` int64 — per-(opp, snapshot)
    #: indices into the global table. ``-1`` is the padding sentinel.
    def_idx: Tensor
    #: ``(n_opponents, n_snapshots, M_def)`` bool — True where
    #: ``def_idx >= 0``.
    def_mask: Tensor

    def save(self, path: Path) -> None:
        """Atomic save: write to ``path.tmp`` then ``rename``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "config": asdict(self.config),
            "global_xy": self.global_xy,
            "global_dates": self.global_dates,
            "global_opponent_idx": self.global_opponent_idx,
            "def_idx": self.def_idx,
            "def_mask": self.def_mask,
        }
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> DefensiveRetrievalCache:
        """Load a cache written by :meth:`save`."""
        d = torch.load(path, weights_only=False)
        cfg_dict = d["config"]
        # JSON serialization turns the anchor_dates tuple into a list;
        # restore the tuple so frozen-dataclass equality holds.
        cfg_dict = {**cfg_dict, "anchor_dates": tuple(cfg_dict["anchor_dates"])}
        cfg = DefensiveRetrievalCacheConfig(**cfg_dict)
        return cls(
            config=cfg,
            global_xy=d["global_xy"],
            global_dates=d["global_dates"],
            global_opponent_idx=d["global_opponent_idx"],
            def_idx=d["def_idx"],
            def_mask=d["def_mask"],
        )


def _vocabulate_defensive(
    shots_df: pd.DataFrame, opp_vocab: OpponentVocab
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter ``shots_df`` to shots whose defending opponent is in the
    vocab and return ``(opp_idx, date_ord, xy)`` arrays sorted ascending
    by ``date_ord``.

    Each row's ``opponent`` column identifies the team the shot was
    attempted *against*, i.e. the defending team, which is the key the
    cache is indexed by. Rows missing ``opponent`` (games at the edge of
    the data where the loader could not resolve both teams) are dropped.

    When ``shots_df`` carries ``team``, a row with ``team == opponent``
    would put a team's own shots into its own defensive pool; such rows
    raise rather than propagate. Without a ``team`` column the check is
    left to the loader.
    """
    if "opponent" not in shots_df.columns:
        raise ValueError(
            "shots_df must carry an 'opponent' column identifying the defending "
            "team for each shot (populated by shotcloud.data.loaders.load_shots)"
        )
    id_to_idx = {oid: i for i, oid in enumerate(opp_vocab.ids)}
    # Cast opponent to string up front to match OpponentVocab's
    # string-normalization convention.
    opp_raw = shots_df["opponent"]
    # Drop rows with missing opponent (NA from boundary games).
    opp_present = opp_raw.notna().to_numpy()
    if not opp_present.all():
        shots_df = shots_df.iloc[opp_present].reset_index(drop=True)
        opp_raw = shots_df["opponent"]
    opp_str = opp_raw.astype(str).to_numpy()
    keep = np.array([o in id_to_idx for o in opp_str], dtype=bool)
    if not keep.all():
        shots_df = shots_df.iloc[keep].reset_index(drop=True)
        opp_str = opp_str[keep]
    # Guard against a team's own shots entering its defensive pool.
    if "team" in shots_df.columns:
        team_str = shots_df["team"].astype(str).to_numpy()
        self_match = team_str == opp_str
        if self_match.any():
            n_bad = int(self_match.sum())
            raise ValueError(
                f"{n_bad} row(s) have team == opponent (e.g. team={team_str[self_match][0]!r}, "
                f"opponent={opp_str[self_match][0]!r}). This violates the load_shots() "
                "contract — a team cannot play itself. Inspect "
                "shotcloud.data.loaders.load_shots._attach_opponent for the source."
            )
    opp_idx = np.array([id_to_idx[o] for o in opp_str], dtype=np.int64)
    date_ord = pd.to_datetime(shots_df["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
    xy = np.stack(
        [
            shots_df["x"].to_numpy(dtype=np.float32),
            shots_df["y"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    order = np.argsort(date_ord, kind="stable")
    return opp_idx[order], date_ord[order], xy[order]


def _build_indices_defensive(
    *,
    opp_idx: np.ndarray,  # (N,) sorted by date — defending opponent for each shot
    date_ord: np.ndarray,  # (N,)
    n_opponents: int,
    anchor_dates: np.ndarray,  # (S,) int64
    config: DefensiveRetrievalCacheConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(opponent, snapshot) defensive retrieval.

    For each opponent ``d`` and anchor ``t_m``, selects the shots
    attempted against ``d`` with date in ``[t_m − window, t_m)`` and keeps
    the ``defensive_support_max`` most recent.

    Returns ``(def_idx, def_mask)`` as numpy arrays; the caller wraps
    them into the :class:`DefensiveRetrievalCache`.
    """
    n_snapshots = anchor_dates.shape[0]
    cap = config.defensive_support_max
    def_idx_out = np.full((n_opponents, n_snapshots, cap), -1, dtype=np.int64)
    def_mask_out = np.zeros((n_opponents, n_snapshots, cap), dtype=bool)
    recency_window = config.defensive_recency_window_days
    for s_idx in range(n_snapshots):
        anchor = int(anchor_dates[s_idx])
        # Causal upper bound: date < anchor.
        i_hi = int(np.searchsorted(date_ord, anchor, side="left"))
        if i_hi == 0:
            continue
        # Recency lower bound: date >= anchor − window.
        i_lo_win = int(np.searchsorted(date_ord, anchor - recency_window, side="left"))
        win_opp = opp_idx[i_lo_win:i_hi]
        win_date = date_ord[i_lo_win:i_hi]
        win_global = np.arange(i_lo_win, i_hi, dtype=np.int64)
        # Group by opponent within the window. For each opp d, take the
        # top-M_def shots by date descending.
        for d_idx in range(n_opponents):
            keep = win_opp == d_idx
            if not keep.any():
                continue
            d_global = win_global[keep]
            d_dates = win_date[keep]
            # argsort by -date gives most-recent-first; stable sort makes
            # ties resolve to lower original index.
            order = np.argsort(-d_dates, kind="stable")
            keep_n = int(min(d_global.size, cap))
            def_idx_out[d_idx, s_idx, :keep_n] = d_global[order[:keep_n]]
            def_mask_out[d_idx, s_idx, :keep_n] = True
    return def_idx_out, def_mask_out


def build_defensive_retrieval_cache(
    *,
    shots_df: pd.DataFrame,
    opp_vocab: OpponentVocab,
    anchor_dates: np.ndarray,
    config: DefensiveRetrievalCacheConfig,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> DefensiveRetrievalCache:
    """Build (or load) a :class:`DefensiveRetrievalCache`.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Must carry ``x, y, player_id, opponent, date`` columns. Rows
        whose ``opponent`` is missing (boundary games) or outside the
        vocab are dropped.
    opp_vocab : OpponentVocab
        Defending-team vocabulary; defines the row indexing of the
        per-(opp, snapshot) tensors.
    anchor_dates : np.ndarray of shape ``(S,)``
        Snapshot anchor dates in **epoch days**.
    config : DefensiveRetrievalCacheConfig
        Retrieval settings; their hash names the cache file.
    cache_dir : Path or None
        If given, the cache is loaded from
        ``cache_dir/defensive_retrieval_cache_{config.config_hash}.pt``
        when that file exists (and ``rebuild=False``), and written
        there after a fresh build. The disk write is atomic
        (``.tmp`` + ``rename``).
    rebuild : bool
        Force a rebuild even when a cached file exists.

    Returns
    -------
    DefensiveRetrievalCache

    Raises
    ------
    ValueError
        If ``anchor_dates`` disagrees with ``config.anchor_dates``, or
        ``shots_df`` lacks an ``opponent`` column or has a row with
        ``team == opponent``.
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
        path = cache_dir / f"defensive_retrieval_cache_{config.config_hash}.pt"
        if path.exists():
            return DefensiveRetrievalCache.load(path)

    opp_idx, date_ord, xy = _vocabulate_defensive(shots_df, opp_vocab)
    def_idx_np, def_mask_np = _build_indices_defensive(
        opp_idx=opp_idx,
        date_ord=date_ord,
        n_opponents=len(opp_vocab),
        anchor_dates=anchor_dates,
        config=config,
    )

    cache = DefensiveRetrievalCache(
        config=config,
        global_xy=torch.from_numpy(xy),
        global_dates=torch.from_numpy(date_ord),
        global_opponent_idx=torch.from_numpy(opp_idx),
        def_idx=torch.from_numpy(def_idx_np),
        def_mask=torch.from_numpy(def_mask_np),
    )

    if cache_dir is not None:
        path = cache_dir / f"defensive_retrieval_cache_{config.config_hash}.pt"
        cache.save(path)

    return cache


def defensive_shots_fingerprint(shots_path: Path) -> str:
    """Return a cheap fingerprint of the shots file for the cache key.

    The fingerprint is ``sha256(path, size, mtime_ns)`` truncated to 16 hex
    characters, the same recipe as
    :func:`shotcloud.models.retrieval_cache.shots_fingerprint`.
    """
    st = shots_path.stat()
    payload: dict[str, Any] = {
        "path": str(shots_path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DEFAULT_DEFENSIVE_RANKING",
    "DEFAULT_DEFENSIVE_RECENCY_WINDOW_DAYS",
    "DEFAULT_DEFENSIVE_SUPPORT_MAX",
    "DefensiveRetrievalCache",
    "DefensiveRetrievalCacheConfig",
    "build_defensive_retrieval_cache",
    "defensive_shots_fingerprint",
]
