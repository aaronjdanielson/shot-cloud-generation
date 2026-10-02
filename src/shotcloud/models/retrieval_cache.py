"""Per-(player, snapshot) causal support retrieval for the AC-KDE backend.

:class:`RetrievalCache` holds, for every (player, snapshot) pair, indices
into a date-sorted global shot table that define the causal support set
used by :class:`~shotcloud.models.retrieval_collaborative_kde.RetrievalCollaborativeKDE`:

* **Own**: the target player's shots dated strictly before the snapshot
  anchor, most recent first, up to ``own_support_max``.
* **Pooled**: shots by other players in the recency window
  ``[anchor − pooled_recency_window_days, anchor)``, ranked by

  ``cos(u_p(t), u_{p_j}(t)) − ln(2) · Δt / half_life``

  (trait cosine similarity plus the log of a half-life recency weight),
  keeping the top ``pooled_support_max``.

Every retained shot predates the anchor date of the snapshot, so the
support is causal by construction. This module is a pure data layer: the
indices are computed once and serialized for reuse by any run with the
same configuration hash.

Determinism notes:

* Candidate ranking uses ``np.argsort(-score, kind="stable")``, not
  ``np.argpartition``, so ties resolve to the lower original index
  reproducibly across NumPy versions.
* The cache file name embeds a config hash that changes whenever any
  retrieval-defining field changes, so a stale cache cannot be
  silently reused under a different config.
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

from shotcloud.training.dataset import PlayerVocab

#: Default support-size caps and retrieval-window settings (also the
#: command-line defaults of the training script).
DEFAULT_OWN_SUPPORT_MAX: int = 1000
DEFAULT_POOLED_SUPPORT_MAX: int = 500
DEFAULT_POOLED_RECENCY_WINDOW_DAYS: int = 180
DEFAULT_POOLED_RECENCY_HALF_LIFE_DAYS: float = 30.0
DEFAULT_SIMILARITY_KIND: str = "cosine_per_snapshot_trait"


@dataclass(frozen=True)
class RetrievalCacheConfig:
    """All retrieval-defining settings that go into the cache hash.

    Two caches with the same ``config_hash`` are guaranteed to have
    identical contents *given the same shots DataFrame and traits
    tensor*. The caller is responsible for keeping the shots fingerprint
    in sync with the actual data file (see :func:`shots_fingerprint`) and
    for setting ``traits_hash`` (see :func:`traits_fingerprint`), since
    pooled candidates are ranked by trait similarity. An empty
    ``traits_hash`` leaves the trait values out of the cache key.
    """

    shots_fingerprint: str
    anchor_dates: tuple[int, ...]
    traits_hash: str = ""
    own_support_max: int = DEFAULT_OWN_SUPPORT_MAX
    pooled_support_max: int = DEFAULT_POOLED_SUPPORT_MAX
    pooled_recency_window_days: int = DEFAULT_POOLED_RECENCY_WINDOW_DAYS
    pooled_recency_half_life_days: float = DEFAULT_POOLED_RECENCY_HALF_LIFE_DAYS
    similarity_kind: str = DEFAULT_SIMILARITY_KIND
    seed: int = 0

    def __post_init__(self) -> None:
        if self.own_support_max <= 0:
            raise ValueError(f"own_support_max must be positive; got {self.own_support_max}")
        if self.pooled_support_max <= 0:
            raise ValueError(f"pooled_support_max must be positive; got {self.pooled_support_max}")
        if self.pooled_recency_window_days <= 0:
            raise ValueError(
                f"pooled_recency_window_days must be positive; "
                f"got {self.pooled_recency_window_days}"
            )
        if self.pooled_recency_half_life_days <= 0.0:
            raise ValueError(
                f"pooled_recency_half_life_days must be positive; "
                f"got {self.pooled_recency_half_life_days}"
            )
        if self.similarity_kind != DEFAULT_SIMILARITY_KIND:
            # Only one similarity is implemented; reject anything else.
            raise ValueError(
                f"unknown similarity_kind={self.similarity_kind!r}; "
                f"only {DEFAULT_SIMILARITY_KIND!r} is implemented"
            )

    @property
    def config_hash(self) -> str:
        """16-character hex SHA-256 of the canonical JSON payload."""
        payload = {
            "shots_fingerprint": self.shots_fingerprint,
            "anchor_dates": list(self.anchor_dates),
            "own_support_max": self.own_support_max,
            "pooled_support_max": self.pooled_support_max,
            "pooled_recency_window_days": self.pooled_recency_window_days,
            "pooled_recency_half_life_days": self.pooled_recency_half_life_days,
            "similarity_kind": self.similarity_kind,
            "seed": self.seed,
        }
        if self.traits_hash:
            payload["traits_hash"] = self.traits_hash
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RetrievalCache:
    """Precomputed retrieval indices and the global shot table they reference.

    ``P`` is the vocabulary size and ``S`` the number of snapshots.
    """

    config: RetrievalCacheConfig
    #: ``(N_global, 2)`` float32 — coordinates of every shot by a
    #: vocabulary player, sorted ascending by ``global_dates``.
    global_xy: Tensor
    #: ``(N_global,)`` int64 — epoch-day date of each global shot.
    global_dates: Tensor
    #: ``(N_global,)`` int64 — vocab index of each global shot's shooter.
    global_shooter_idx: Tensor
    #: ``(P, S, own_support_max)`` int64 — own-history indices into the
    #: global table. ``-1`` is the padding sentinel.
    own_idx: Tensor
    #: ``(P, S, own_support_max)`` bool — True where ``own_idx >= 0``.
    own_mask: Tensor
    #: ``(P, S, pooled_support_max)`` int64 — pooled-retrieval indices
    #: into the global table, sorted by ``cos_sim + log_recency``
    #: descending. ``-1`` is the padding sentinel.
    pooled_idx: Tensor
    #: ``(P, S, pooled_support_max)`` bool — True where ``pooled_idx >= 0``.
    pooled_mask: Tensor

    def save(self, path: Path) -> None:
        """Atomic save: write to ``path.tmp`` then ``rename``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "config": asdict(self.config),
            "global_xy": self.global_xy,
            "global_dates": self.global_dates,
            "global_shooter_idx": self.global_shooter_idx,
            "own_idx": self.own_idx,
            "own_mask": self.own_mask,
            "pooled_idx": self.pooled_idx,
            "pooled_mask": self.pooled_mask,
        }
        torch.save(payload, tmp)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RetrievalCache:
        """Load a cache written by :meth:`save`."""
        d = torch.load(path, weights_only=False)
        cfg_dict = d["config"]
        # JSON serialization turns the anchor_dates tuple into a list;
        # restore the tuple so frozen-dataclass equality holds.
        cfg_dict = {**cfg_dict, "anchor_dates": tuple(cfg_dict["anchor_dates"])}
        cfg = RetrievalCacheConfig(**cfg_dict)
        return cls(
            config=cfg,
            global_xy=d["global_xy"],
            global_dates=d["global_dates"],
            global_shooter_idx=d["global_shooter_idx"],
            own_idx=d["own_idx"],
            own_mask=d["own_mask"],
            pooled_idx=d["pooled_idx"],
            pooled_mask=d["pooled_mask"],
        )


def _vocabulate(
    shots_df: pd.DataFrame, player_vocab: PlayerVocab
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter ``shots_df`` to shots whose shooter is in the vocab and
    return ``(shooter_idx, date_ord, xy)`` arrays sorted ascending by
    ``date_ord``."""
    id_to_idx = {pid: i for i, pid in enumerate(player_vocab.ids)}
    pid_str = shots_df["player_id"].astype(str).to_numpy()
    keep = np.array([p in id_to_idx for p in pid_str], dtype=bool)
    if not keep.all():
        shots_df = shots_df.iloc[keep].reset_index(drop=True)
        pid_str = pid_str[keep]
    shooter_idx = np.array([id_to_idx[p] for p in pid_str], dtype=np.int64)
    date_ord = pd.to_datetime(shots_df["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
    xy = np.stack(
        [
            shots_df["x"].to_numpy(dtype=np.float32),
            shots_df["y"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    order = np.argsort(date_ord, kind="stable")
    return shooter_idx[order], date_ord[order], xy[order]


def _build_indices(
    *,
    shooter_idx: np.ndarray,  # (N,) sorted by date
    date_ord: np.ndarray,  # (N,)
    n_players: int,
    anchor_dates: np.ndarray,  # (S,) int64
    traits: Tensor,  # (P, S, T)
    config: RetrievalCacheConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the retrieval for every (player, snapshot).

    Returns ``(own_idx, own_mask, pooled_idx, pooled_mask)`` as NumPy arrays.
    """
    n_snapshots = anchor_dates.shape[0]
    own_idx = np.full((n_players, n_snapshots, config.own_support_max), -1, dtype=np.int64)
    own_mask = np.zeros((n_players, n_snapshots, config.own_support_max), dtype=bool)
    pooled_idx = np.full((n_players, n_snapshots, config.pooled_support_max), -1, dtype=np.int64)
    pooled_mask = np.zeros((n_players, n_snapshots, config.pooled_support_max), dtype=bool)
    traits_np = traits.detach().cpu().numpy()
    recency_window = config.pooled_recency_window_days
    half_life = config.pooled_recency_half_life_days
    log2_per_day = float(np.log(2.0)) / half_life
    for s_idx in range(n_snapshots):
        anchor = int(anchor_dates[s_idx])
        i_hi = int(np.searchsorted(date_ord, anchor, side="left"))  # date < anchor
        if i_hi == 0:
            continue
        i_lo_win = int(np.searchsorted(date_ord, anchor - recency_window, side="left"))
        causal_shooter = shooter_idx[:i_hi]
        causal_date = date_ord[:i_hi]
        win_shooter = shooter_idx[i_lo_win:i_hi]
        win_date = date_ord[i_lo_win:i_hi]
        win_global = np.arange(i_lo_win, i_hi, dtype=np.int64)
        # Cosine sim per shooter at this snapshot.
        all_traits = traits_np[:, s_idx, :]
        norms = np.linalg.norm(all_traits, axis=-1)
        eps = 1e-12
        for p_idx in range(n_players):
            target_trait = all_traits[p_idx]
            denom = norms * (norms[p_idx] + eps) + eps
            cos_sim = (all_traits @ target_trait) / denom  # (P,)
            # Own — target's causal shots, most-recent-first.
            own_global = np.flatnonzero(causal_shooter == p_idx)
            if own_global.size > 0:
                own_dates_p = causal_date[own_global]
                # argsort by negative date is stable and gives most-recent first;
                # ties resolve to lower original index.
                order = np.argsort(-own_dates_p, kind="stable")
                keep_n = int(min(own_global.size, config.own_support_max))
                own_idx[p_idx, s_idx, :keep_n] = own_global[order[:keep_n]]
                own_mask[p_idx, s_idx, :keep_n] = True
            # Pooled — in window, not target, scored by cos_sim + log_recency.
            if win_shooter.size == 0:
                continue
            pop_keep = win_shooter != p_idx
            if not pop_keep.any():
                continue
            pop_shooter = win_shooter[pop_keep]
            pop_date = win_date[pop_keep]
            pop_global = win_global[pop_keep]
            sim = cos_sim[pop_shooter]
            log_rec = -((anchor - pop_date).astype(np.float64)) * log2_per_day
            score = sim.astype(np.float64) + log_rec
            order = np.argsort(-score, kind="stable")
            keep_n = int(min(score.size, config.pooled_support_max))
            pooled_idx[p_idx, s_idx, :keep_n] = pop_global[order[:keep_n]]
            pooled_mask[p_idx, s_idx, :keep_n] = True
    return own_idx, own_mask, pooled_idx, pooled_mask


def build_retrieval_cache(
    *,
    shots_df: pd.DataFrame,
    player_vocab: PlayerVocab,
    anchor_dates: np.ndarray,
    traits: Tensor,
    config: RetrievalCacheConfig,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> RetrievalCache:
    """Build (or load) a :class:`RetrievalCache`.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Must carry ``x, y, player_id, date`` columns. Shots whose
        ``player_id`` is not in ``player_vocab`` are dropped.
    player_vocab : PlayerVocab
        The model's player vocabulary; defines the row indexing of
        the per-(player, snapshot) tensors.
    anchor_dates : np.ndarray of shape ``(S,)``
        Snapshot anchor dates in **epoch days**; must equal
        ``config.anchor_dates``.
    traits : Tensor of shape ``(P, S, T)``
        Per-snapshot causal player-trait vectors; pooled candidates are
        ranked by cosine similarity in this space. When
        ``config.traits_hash`` is set it must equal
        ``traits_fingerprint(traits)``.
    config : RetrievalCacheConfig
        Retrieval settings; their hash names the cache file.
    cache_dir : Path or None
        If given, the cache is loaded from
        ``cache_dir/retrieval_cache_{config.config_hash}.pt`` when
        that file exists (and ``rebuild=False``), and written there
        after a fresh build. The disk write is atomic
        (``.tmp`` + ``rename``).
    rebuild : bool
        Force a rebuild even when a cached file exists at the
        expected path.

    Returns
    -------
    RetrievalCache

    Raises
    ------
    ValueError
        If ``anchor_dates`` disagrees with ``config.anchor_dates``,
        ``traits`` has the wrong shape, or ``config.traits_hash`` is set
        and does not match ``traits``.
    """
    if anchor_dates.ndim != 1:
        raise ValueError(f"anchor_dates must be 1D; got shape {anchor_dates.shape}")
    if anchor_dates.dtype != np.int64:
        anchor_dates = anchor_dates.astype(np.int64)
    n_snapshots = anchor_dates.shape[0]
    if tuple(config.anchor_dates) != tuple(int(d) for d in anchor_dates):
        raise ValueError(
            "config.anchor_dates must equal the anchor_dates argument "
            "(the hash depends on them; mismatch would corrupt the cache key)"
        )
    if traits.dim() != 3 or traits.shape[1] != n_snapshots:
        raise ValueError(f"traits must be (P, S={n_snapshots}, T); got {tuple(traits.shape)}")

    n_players = len(player_vocab)
    if traits.shape[0] != n_players:
        raise ValueError(
            f"traits.shape[0]={traits.shape[0]} must equal len(player_vocab)={n_players}"
        )

    if config.traits_hash and config.traits_hash != traits_fingerprint(traits):
        raise ValueError(
            "config.traits_hash does not match traits "
            "(the hash depends on it; mismatch would corrupt the cache key)"
        )

    if cache_dir is not None and not rebuild:
        path = cache_dir / f"retrieval_cache_{config.config_hash}.pt"
        if path.exists():
            return RetrievalCache.load(path)

    shooter_idx, date_ord, xy = _vocabulate(shots_df, player_vocab)
    own_idx_np, own_mask_np, pooled_idx_np, pooled_mask_np = _build_indices(
        shooter_idx=shooter_idx,
        date_ord=date_ord,
        n_players=n_players,
        anchor_dates=anchor_dates,
        traits=traits,
        config=config,
    )

    cache = RetrievalCache(
        config=config,
        global_xy=torch.from_numpy(xy),
        global_dates=torch.from_numpy(date_ord),
        global_shooter_idx=torch.from_numpy(shooter_idx),
        own_idx=torch.from_numpy(own_idx_np),
        own_mask=torch.from_numpy(own_mask_np),
        pooled_idx=torch.from_numpy(pooled_idx_np),
        pooled_mask=torch.from_numpy(pooled_mask_np),
    )

    if cache_dir is not None:
        path = cache_dir / f"retrieval_cache_{config.config_hash}.pt"
        cache.save(path)

    return cache


def traits_fingerprint(traits: Tensor | np.ndarray) -> str:
    """Return a 16-character hex SHA-256 of the trait values for the cache key.

    Used as :attr:`RetrievalCacheConfig.traits_hash`, so a cache built from
    one trait table is never reused with another.
    """
    values = traits.detach().cpu().numpy() if isinstance(traits, Tensor) else traits
    arr = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def shots_fingerprint(shots_path: Path) -> str:
    """Return a cheap fingerprint of the shots file for the cache key.

    The fingerprint is ``sha256(path, size, mtime_ns)`` truncated to 16 hex
    characters, used as :attr:`RetrievalCacheConfig.shots_fingerprint`.
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
    "DEFAULT_OWN_SUPPORT_MAX",
    "DEFAULT_POOLED_RECENCY_HALF_LIFE_DAYS",
    "DEFAULT_POOLED_RECENCY_WINDOW_DAYS",
    "DEFAULT_POOLED_SUPPORT_MAX",
    "DEFAULT_SIMILARITY_KIND",
    "RetrievalCache",
    "RetrievalCacheConfig",
    "build_retrieval_cache",
    "shots_fingerprint",
    "traits_fingerprint",
]
