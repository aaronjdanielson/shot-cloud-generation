"""Dataset for joint Gibbs training of the marked point-process model.

:class:`GibbsShotDataset` materializes the per-shot tensors that the
joint trainer consumes:

* ``player_idx`` into :class:`~shotcloud.training.PlayerVocab`,
* ``opp_idx`` into :class:`~shotcloud.training.OpponentVocab`,
* ``snapshot_idx`` from
  :meth:`SnapshotStore.get_snapshot_index(date)` — required by
  :class:`~shotcloud.models.AdaptiveOffensivePrior` and
  :class:`~shotcloud.models.AdaptiveDefensiveField` for their causal
  date masks,
* ``cell_idx`` — the observed court cell,
* ``tau_bin`` — minute bin in ``[0, 48)`` (regulation; OT folded into
  the final bin) consumed by
  :class:`~shotcloud.models.TimingSoftmaxHead`,
* ``x_n_raw`` of shape ``(CONTEXT_DIM,)`` — the canonical context
  vector that :class:`~shotcloud.models.ContextMLP` will transform
  into ``x_n`` inside the trainer,
* ``game_id`` index into the dataset's per-game table.

A parallel :attr:`per_game` table maps ``game_id`` to
``(x_n_raw_game, K_obs)`` for the negative-binomial count loss. The
count loss is computed once per game, not once per shot.

The dataset does not own any KDEs or models; it only routes
indices and features. The trainer composes the spatial / count /
timing modules and consumes these tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from shotcloud.data.context import CONTEXT_DIM, ContextEncoder
from shotcloud.data.prior_outcomes import PRIOR_OUTCOME_DIM, compute_prior_outcome_features
from shotcloud.data.snapshots import SnapshotStore
from shotcloud.data.within_game_history import (
    MAX_PRIOR_SHOTS,
    WITHIN_GAME_DIM,
    WITHIN_GAME_SEQ_DIM,
    compute_within_game_features,
    compute_within_game_sequence,
)
from shotcloud.grids import CourtGrid
from shotcloud.training.dataset import OpponentVocab, PlayerVocab

#: Number of regulation minutes; OT shots fold into the final bin.
N_TIMING_BINS: int = 48


def _compute_tau_bin(
    period: NDArray[np.int64], time_remaining_sec: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Map shot timing → tau_bin in [0, 48).

    The loader's ``time_remaining_sec`` column is, despite its name,
    the **total seconds elapsed in the game** (built from
    ``MINUTES_REMAINING`` + ``SECONDS_REMAINING`` + ``PERIOD``; see
    :func:`shotcloud.data.loaders.load_shots`). The bin index is
    therefore ``floor(time_remaining_sec / 60)``, clipped to
    ``[0, 47]`` so overtime shots (game minute > 47) fold into the
    final bin.

    The ``period`` argument is unused (it's implicit in the elapsed
    seconds) but kept in the signature for caller-side legibility
    and forward compatibility if the schema is ever renamed.
    """
    del period  # see docstring
    bin_idx = np.floor(time_remaining_sec / 60.0).astype(np.int64)
    clipped: NDArray[np.int64] = np.clip(bin_idx, 0, N_TIMING_BINS - 1).astype(np.int64)
    return clipped


@dataclass(frozen=True)
class PerGameTable:
    """Per-game targets and context for the count factor.

    Aligned across ``game_id``: row ``i`` corresponds to the game with
    integer id ``i`` (assigned by :class:`GibbsShotDataset`).
    """

    x_n_raw: Tensor  # (n_games, CONTEXT_DIM)
    k_obs: Tensor  # (n_games,) int64 shot count


#: Per-shot batch tuple emitted by :class:`GibbsShotDataset.__getitem__`:
#: ``(player_idx, opp_idx, snapshot_idx, cell_idx, tau_bin, x_n_raw,
#: game_idx, shot_xy, h_within_game, prior_outcome, prior_seq,
#: prior_lengths)`` — 12 per-shot tensors.
_BatchTuple = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]


class GibbsShotDataset(Dataset[_BatchTuple]):
    """Per-shot training examples for joint Gibbs training.

    Parameters
    ----------
    shots_df : DataFrame
        Canonical-schema shot rows with at least ``x``, ``y``,
        ``player_id``, ``date``, ``opponent``, ``period``,
        ``time_remaining_sec``, ``game_id``.
    snapshot_store : SnapshotStore
        Frozen causal store. Provides
        :meth:`get_snapshot_index(date)` for every shot.
    grid : CourtGrid
        Discretization shared with the snapshot bundles.
    player_vocab : PlayerVocab
        Players covered by the offensive prior; shots whose player
        is not in the vocab are dropped.
    opp_vocab : OpponentVocab or None
        Required when the trainer will consume the defensive field;
        when ``None`` the dataset emits sentinel zeros for
        ``opp_idx``. The downstream :class:`ConditionalGibbsDecoder`
        is offense-only-safe in this mode if its defensive field is
        configured to return uniform feasibility.
    context_encoder : ContextEncoder
        Transforms the shots frame to the canonical 27-D context
        vector. Required (no sentinel mode).
    dtype : torch.dtype, default ``torch.float32``
    """

    def __init__(
        self,
        shots_df: pd.DataFrame,
        snapshot_store: SnapshotStore,
        grid: CourtGrid,
        player_vocab: PlayerVocab,
        opp_vocab: OpponentVocab | None,
        context_encoder: ContextEncoder,
        dtype: torch.dtype = torch.float32,
        with_spatial_hawkes_residual: bool = False,
        spatial_hawkes_sigma_ft: float = 4.0,
    ) -> None:
        required = ("x", "y", "player_id", "date", "period", "time_remaining_sec", "game_id")
        for col in required:
            if col not in shots_df.columns:
                raise KeyError(f"shots_df must contain column {col!r}")
        if opp_vocab is not None and "opponent" not in shots_df.columns:
            raise KeyError("opp_vocab provided but shots_df has no 'opponent' column")

        # Filter to in-vocab players and (when present) opponents.
        player_keys = {str(pid) for pid in player_vocab.ids}
        df = shots_df[shots_df["player_id"].astype(str).isin(player_keys)].copy()
        if opp_vocab is not None:
            opp_keys = {str(o) for o in opp_vocab.ids}
            df = df[df["opponent"].astype(str).isin(opp_keys)].copy()

        # Drop out-of-court shots before computing any indices.
        x_coord = df["x"].to_numpy(dtype=np.float64)
        y_coord = df["y"].to_numpy(dtype=np.float64)
        cells = grid.coord_to_cell(x_coord, y_coord)
        valid = cells >= 0
        df = df.loc[valid].reset_index(drop=True)
        cells = cells[valid].astype(np.int64)
        # Exact continuous shot coords parallel to the cell index. Kept
        # as float32 — the continuous-coordinate spatial loss uses these
        # directly (not the cell-center snap) so geometric distance to
        # the predicted distribution is preserved.
        shot_xy_np = np.stack([x_coord[valid], y_coord[valid]], axis=1, dtype=np.float32)
        if len(df) == 0:
            raise ValueError("no in-court shots remained after filtering")

        # Compute per-shot snapshot_idx in one searchsorted call. Shots
        # whose date precedes the first anchor (searchsorted returns 0,
        # decremented to -1) have no valid bundle; drop them to keep
        # the causal contract intact.
        dates = np.asarray(df["date"], dtype="datetime64[D]")
        anchor_dates = snapshot_store.anchor_dates
        snap_idx_np = np.searchsorted(anchor_dates, dates, side="right").astype(np.int64) - 1
        has_snap = snap_idx_np >= 0
        if not has_snap.any():
            raise ValueError("no shots fall on or after the first snapshot anchor")
        df = df.loc[has_snap].reset_index(drop=True)
        cells = cells[has_snap]
        snap_idx_np = snap_idx_np[has_snap]
        shot_xy_np = shot_xy_np[has_snap]

        # Per-shot indices.
        player_idx_np = np.array([player_vocab.to_idx(p) for p in df["player_id"]], dtype=np.int64)
        if opp_vocab is not None:
            opp_idx_np = np.array([opp_vocab.to_idx(o) for o in df["opponent"]], dtype=np.int64)
        else:
            opp_idx_np = np.zeros(len(df), dtype=np.int64)

        period_np = df["period"].to_numpy(dtype=np.int64)
        time_rem_np = df["time_remaining_sec"].to_numpy(dtype=np.float64)
        tau_bin_np = _compute_tau_bin(period_np, time_rem_np)

        # Canonical context features for every shot.
        ctx = context_encoder.transform(df)
        if ctx.shape != (len(df), CONTEXT_DIM):
            raise ValueError(
                f"context_encoder.transform produced {ctx.shape}; expected "
                f"({len(df)}, {CONTEXT_DIM})"
            )

        # Within-game causal shot-history features h_{n,r}. The
        # featurizer is causal by construction; built on the
        # post-filter DataFrame so the row order is aligned with
        # cells / ctx / shot_xy.
        h_within_game_np = compute_within_game_features(df)
        if h_within_game_np.shape != (len(df), WITHIN_GAME_DIM):
            raise ValueError(
                f"compute_within_game_features produced {h_within_game_np.shape}; "
                f"expected ({len(df)}, {WITHIN_GAME_DIM})"
            )

        # Causal prior-outcome summary o_{n,r} (Phase 2 of the
        # 2026-06-07 audit). Computed when the input carries a
        # ``made`` column; zero-filled otherwise so the residual
        # encoder can be wired with ``outcome_dim > 0`` without
        # forcing every caller to supply outcomes.
        if "made" in df.columns:
            prior_outcome_np = compute_prior_outcome_features(df)
        else:
            prior_outcome_np = np.zeros((len(df), PRIOR_OUTCOME_DIM), dtype=np.float32)

        # Phase 1 B1 (2026-06-09): when the spatial-Hawkes flag is
        # set, append the 8-dim causal prior-shot KDE feature to
        # ``o_{n,r}``. The residual encoder's ``outcome_dim`` then
        # bumps from 9 to 17; the zero-init projection invariant
        # carries through unchanged.
        if with_spatial_hawkes_residual:
            from shotcloud.data.prior_shot_kde import (
                PRIOR_SHOT_KDE_DIM,
                compute_prior_shot_kde_features,
            )

            spatial_hawkes_np = compute_prior_shot_kde_features(
                df, sigma_h_ft=spatial_hawkes_sigma_ft
            )
            if spatial_hawkes_np.shape != (len(df), PRIOR_SHOT_KDE_DIM):
                raise ValueError(
                    f"compute_prior_shot_kde_features produced {spatial_hawkes_np.shape}; "
                    f"expected ({len(df)}, {PRIOR_SHOT_KDE_DIM})"
                )
            prior_outcome_np = np.concatenate([prior_outcome_np, spatial_hawkes_np], axis=-1)
            expected_outcome_dim = PRIOR_OUTCOME_DIM + PRIOR_SHOT_KDE_DIM
        else:
            expected_outcome_dim = PRIOR_OUTCOME_DIM

        if prior_outcome_np.shape != (len(df), expected_outcome_dim):
            raise ValueError(
                f"compute_prior_outcome_features (+ spatial hawkes) produced "
                f"{prior_outcome_np.shape}; expected ({len(df)}, {expected_outcome_dim})"
            )

        # G1 within-game shot sequence (paper §10). Computed lazily here
        # so the per-shot prior-sequence tensor + per-shot length are
        # available to the trainer whether or not the G1 GRU is wired;
        # the per-row memory is bounded by MAX_PRIOR_SHOTS and the
        # tensors are zero-shaped when the dataset is empty.
        prior_seq_np, prior_lengths_np = compute_within_game_sequence(df)
        expected_seq_shape = (len(df), MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)
        if prior_seq_np.shape != expected_seq_shape:
            raise ValueError(
                f"compute_within_game_sequence produced {prior_seq_np.shape}; "
                f"expected {expected_seq_shape}"
            )
        if prior_lengths_np.shape != (len(df),):
            raise ValueError(
                f"compute_within_game_sequence lengths shape {prior_lengths_np.shape}; "
                f"expected ({len(df)},)"
            )

        # Per-game integer index. Each game is associated with one
        # (player_id, game_id) pair — a player taking shots in different
        # games has a different count target per game. pd.factorize
        # over the combined string key assigns 0..n_games-1 in order
        # of first appearance, identical to the previous dict-build loop.
        #
        # CRITICAL: the separator must not be a null byte. In pandas 3.x,
        # ``pd.factorize`` on a numpy object array containing null bytes
        # uses a C-string hash that truncates at the null, collapsing
        # every game of a given player into a single ``game_idx`` —
        # verified 2026-06-08 on pandas 3.0.2 / numpy 2.4.4. Using ``|``
        # as the separator is safe because neither ``player_id`` nor
        # ``game_id`` ever contains it. See
        # ``tests/test_training_gibbs_dataset_factorize.py`` for the
        # regression check.
        game_keys = (df["player_id"].astype(str) + "|" + df["game_id"].astype(str)).to_numpy()
        game_id_np, _ = pd.factorize(game_keys, sort=False)
        game_id_np = game_id_np.astype(np.int64)
        n_games = int(game_id_np.max()) + 1
        k_obs_np = np.bincount(game_id_np, minlength=n_games).astype(np.int64)

        # Per-game x_n: take the context of the game's first shot —
        # all shots within a player-game share the same x_n by
        # construction (pregame state is per-player-game). Stable
        # argsort by game_idx; the first occurrence of each game is
        # marked by a change in the sorted key.
        order = np.argsort(game_id_np, kind="stable")
        sorted_games = game_id_np[order]
        first_in_sorted = np.empty(n_games, dtype=np.int64)
        first_in_sorted[0] = 0
        first_in_sorted[1:] = np.flatnonzero(np.diff(sorted_games)) + 1
        first_shot_idx_per_game = order[first_in_sorted]
        x_n_raw_game = ctx[first_shot_idx_per_game]

        # Stash tensors.
        self.player_vocab = player_vocab
        self.opp_vocab = opp_vocab
        self.snapshot_store = snapshot_store
        self.player_idx: Tensor = torch.from_numpy(player_idx_np)
        self.opp_idx: Tensor = torch.from_numpy(opp_idx_np)
        self.snapshot_idx: Tensor = torch.from_numpy(snap_idx_np)
        self.cell_idx: Tensor = torch.from_numpy(cells)
        self.tau_bin: Tensor = torch.from_numpy(tau_bin_np)
        self.x_n_raw: Tensor = torch.from_numpy(ctx).to(dtype=dtype)
        self.game_idx: Tensor = torch.from_numpy(game_id_np)
        self.shot_xy: Tensor = torch.from_numpy(shot_xy_np).to(dtype=dtype)
        self.h_within_game: Tensor = torch.from_numpy(h_within_game_np).to(dtype=dtype)
        self.prior_outcome: Tensor = torch.from_numpy(prior_outcome_np).to(dtype=dtype)
        # Exposes the realized outcome-feature dim so callers building a
        # model can size the residual encoder's outcome branch directly.
        # PRIOR_OUTCOME_DIM (=9) when no spatial-Hawkes flag is set;
        # PRIOR_OUTCOME_DIM + PRIOR_SHOT_KDE_DIM (=17) when on.
        self.outcome_feature_dim: int = int(prior_outcome_np.shape[-1])
        self.with_spatial_hawkes_residual: bool = bool(with_spatial_hawkes_residual)
        self.spatial_hawkes_sigma_ft: float = float(spatial_hawkes_sigma_ft)
        self.prior_seq: Tensor = torch.from_numpy(prior_seq_np).to(dtype=dtype)
        self.prior_lengths: Tensor = torch.from_numpy(prior_lengths_np)
        self.per_game: PerGameTable = PerGameTable(
            x_n_raw=torch.from_numpy(x_n_raw_game).to(dtype=dtype),
            k_obs=torch.from_numpy(k_obs_np),
        )
        # Post-filter per-shot metadata. The dataset's filtering chain
        # (vocab membership, in-court coordinates, snapshot validity) is
        # opaque to external callers; this small DataFrame lets the
        # presence/timing pipeline align on (player_id, date, starter)
        # exactly to the dataset's row order without replicating filters.
        meta_cols = ["player_id", "date"]
        if "starter" in df.columns:
            meta_cols.append("starter")
        self.per_shot_meta: pd.DataFrame = df[meta_cols].reset_index(drop=True)

    @property
    def n_shots(self) -> int:
        return int(self.cell_idx.shape[0])

    @property
    def n_games(self) -> int:
        return int(self.per_game.k_obs.shape[0])

    @property
    def has_opponents(self) -> bool:
        return self.opp_vocab is not None

    def __len__(self) -> int:
        return self.n_shots

    def __getitem__(self, idx: int) -> _BatchTuple:
        return (
            self.player_idx[idx],
            self.opp_idx[idx],
            self.snapshot_idx[idx],
            self.cell_idx[idx],
            self.tau_bin[idx],
            self.x_n_raw[idx],
            self.game_idx[idx],
            self.shot_xy[idx],
            self.h_within_game[idx],
            self.prior_outcome[idx],
            self.prior_seq[idx],
            self.prior_lengths[idx],
        )

    def subset(self, n_shots: int, *, seed: int = 0) -> GibbsShotDataset:
        """Return a deterministic-random sub-dataset of ``n_shots`` rows.

        For the fast validation ladder: train a small subset to test
        learning signal before committing compute to a full epoch.

        The per-game table is **re-derived** for the sampled rows
        because the count loss is amortized over per-game shot counts,
        and the original ``per_game.k_obs`` would over-count for any
        partially-sampled game.

        ``n_shots >= self.n_shots`` returns ``self`` unchanged (cheap
        no-op).
        """
        if n_shots <= 0:
            raise ValueError(f"n_shots must be > 0; got {n_shots}")
        if n_shots >= self.n_shots:
            return self

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(self.n_shots, size=n_shots, replace=False)).astype(np.int64)
        keep_t = torch.from_numpy(keep)

        # Build a shallow copy by bypassing __init__ — the source
        # tensors are already validated.
        out = object.__new__(type(self))
        out.player_vocab = self.player_vocab
        out.opp_vocab = self.opp_vocab
        out.snapshot_store = self.snapshot_store
        out.player_idx = self.player_idx[keep_t]
        out.opp_idx = self.opp_idx[keep_t]
        out.snapshot_idx = self.snapshot_idx[keep_t]
        out.cell_idx = self.cell_idx[keep_t]
        out.tau_bin = self.tau_bin[keep_t]
        out.x_n_raw = self.x_n_raw[keep_t]
        out.shot_xy = self.shot_xy[keep_t]
        out.h_within_game = self.h_within_game[keep_t]
        out.prior_outcome = self.prior_outcome[keep_t]
        out.outcome_feature_dim = self.outcome_feature_dim
        out.with_spatial_hawkes_residual = self.with_spatial_hawkes_residual
        out.spatial_hawkes_sigma_ft = self.spatial_hawkes_sigma_ft
        out.prior_seq = self.prior_seq[keep_t]
        out.prior_lengths = self.prior_lengths[keep_t]

        # Re-derive game_idx + per_game table over the sampled rows.
        # ``torch.unique(return_inverse=True)`` collapses original
        # game ids to contiguous [0..n_kept_games) and gives the
        # remapped per-shot idx in one shot.
        orig_game = self.game_idx[keep_t]
        kept_game_ids, remapped_game = torch.unique(orig_game, return_inverse=True)
        out.game_idx = remapped_game.to(torch.int64)
        n_kept_games = int(kept_game_ids.shape[0])

        # Per-game x_n_raw: first shot of each game in the sampled
        # rows. argsort by game id + take first occurrence per group.
        order = torch.argsort(out.game_idx, stable=True)
        sorted_games = out.game_idx[order]
        first_in_sorted = torch.cat(
            [
                torch.tensor([0], dtype=torch.int64),
                torch.nonzero(sorted_games[1:] != sorted_games[:-1], as_tuple=False).flatten() + 1,
            ]
        )
        first_shot_per_game = order[first_in_sorted]
        per_game_x = out.x_n_raw[first_shot_per_game]
        per_game_k = torch.bincount(out.game_idx, minlength=n_kept_games).to(torch.int64)
        out.per_game = PerGameTable(x_n_raw=per_game_x, k_obs=per_game_k)
        return out
