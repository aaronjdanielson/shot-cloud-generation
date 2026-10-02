"""Per-shot dataset for joint training of the marked point-process model.

:class:`GibbsShotDataset` materializes, for every in-court shot, the
tensors consumed by :func:`~shotcloud.training.train_gibbs`:

* ``player_idx`` -- index into :class:`~shotcloud.training.PlayerVocab`;
* ``opp_idx`` -- index into :class:`~shotcloud.training.OpponentVocab`
  (zero when no opponent vocabulary is given);
* ``snapshot_idx`` -- index of the latest
  :class:`~shotcloud.data.snapshots.SnapshotStore` anchor on or before
  the shot's date, which selects the causal snapshot data used by the
  support backend and the defensive features;
* ``cell_idx`` -- the observed court cell (used by the grid-cell losses);
* ``tau_bin`` -- game-minute bin in ``[0, 48)`` with overtime folded into
  the final bin, the target of
  :class:`~shotcloud.models.TimingSoftmaxHead`;
* ``x_n_raw`` -- the raw pregame context vector of length ``CONTEXT_DIM``,
  mapped to ``x_n`` by :class:`~shotcloud.models.ContextMLP` inside the
  trainer;
* ``game_idx`` -- index into the per-game table;
* ``shot_xy`` -- the exact shot coordinates in feet;
* ``h_within_game`` -- causal summary of the player's earlier shots in
  the same game;
* ``prior_outcome`` -- causal summary of the outcomes of those earlier
  shots (optionally extended by a prior-shot KDE feature);
* ``prior_seq``, ``prior_lengths`` -- the padded sequence of earlier
  same-game shots and its length, for an optional within-game recurrent
  encoder.

A parallel :attr:`GibbsShotDataset.per_game` table holds the pregame
context and observed shot count ``K_obs`` of every player-game, so the
negative-binomial count loss is evaluated once per game rather than once
per shot.

The dataset owns no models; it only computes indices and features. All
features use information available before the shot: snapshot data from
earlier games, and within-game features from earlier shots of the same
game.
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
    """Map shot timing to a game-minute bin in ``[0, 48)``.

    The loader's ``time_remaining_sec`` column holds, despite its name,
    the total seconds elapsed in the game (derived from ``PERIOD``,
    ``MINUTES_REMAINING`` and ``SECONDS_REMAINING``; see
    :func:`shotcloud.data.loaders.load_shots`). The bin index is
    ``floor(time_remaining_sec / 60)`` clipped to ``[0, 47]``, so
    overtime shots fold into the final bin.

    ``period`` is unused, since it is implied by the elapsed seconds.
    """
    del period  # implied by the elapsed seconds
    bin_idx = np.floor(time_remaining_sec / 60.0).astype(np.int64)
    clipped: NDArray[np.int64] = np.clip(bin_idx, 0, N_TIMING_BINS - 1).astype(np.int64)
    return clipped


@dataclass(frozen=True)
class PerGameTable:
    """Per-game targets and context for the count factor.

    Row ``i`` is the player-game with ``game_idx == i`` as assigned by
    :class:`GibbsShotDataset`.
    """

    x_n_raw: Tensor  # (n_games, CONTEXT_DIM)
    k_obs: Tensor  # (n_games,) int64 shot count


#: Per-shot batch tuple emitted by :class:`GibbsShotDataset.__getitem__`:
#: ``(player_idx, opp_idx, snapshot_idx, cell_idx, tau_bin, x_n_raw,
#: game_idx, shot_xy, h_within_game, prior_outcome, prior_seq,
#: prior_lengths)``.
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
    """Per-shot training examples for :func:`~shotcloud.training.train_gibbs`.

    Shots are dropped when the player (or, with ``opp_vocab``, the
    opponent) is not in the vocabulary, when the coordinates fall off the
    grid, or when the shot's date precedes the first snapshot anchor (no
    causal snapshot exists for it). Each item is the tuple of per-shot
    tensors described in the module docstring.

    Parameters
    ----------
    shots_df : DataFrame
        Canonical-schema shot rows with at least ``x``, ``y``,
        ``player_id``, ``date``, ``period``, ``time_remaining_sec`` and
        ``game_id``, plus ``opponent`` when ``opp_vocab`` is given and the
        columns ``context_encoder`` requires. An optional ``made`` column
        enables the prior-outcome features; without it they are zero.
    snapshot_store : SnapshotStore
        Causal snapshot store whose ``anchor_dates`` define
        ``snapshot_idx``.
    grid : CourtGrid
        Court discretization used for ``cell_idx`` and the on-court filter.
    player_vocab : PlayerVocab
        Players covered by the model.
    opp_vocab : OpponentVocab or None
        Required by opponent-conditioned components. When ``None``,
        ``opp_idx`` is all zeros.
    context_encoder : ContextEncoder
        Fitted encoder mapping the shot rows to ``(n, CONTEXT_DIM)``
        context vectors.
    dtype : torch.dtype, default ``torch.float32``
        Floating dtype of the feature tensors.
    with_spatial_hawkes_residual : bool, default False
        Append the causal prior-shot KDE feature
        (:func:`~shotcloud.data.prior_shot_kde.compute_prior_shot_kde_features`)
        to ``prior_outcome``, widening it from ``PRIOR_OUTCOME_DIM`` to
        ``PRIOR_OUTCOME_DIM + PRIOR_SHOT_KDE_DIM``.
    spatial_hawkes_sigma_ft : float, default 4.0
        Kernel bandwidth in feet of the prior-shot KDE feature.

    Attributes
    ----------
    per_game : PerGameTable
        Pregame context and observed shot count per player-game.
    outcome_feature_dim : int
        Width of ``prior_outcome``, for sizing the residual encoder's
        outcome branch.
    per_shot_meta : DataFrame
        ``player_id``, ``date`` and (when present) ``starter`` of the
        retained shots, in dataset row order.

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If no shots survive filtering or a feature builder returns an
        unexpected shape.
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
        # Exact shot coordinates, parallel to the cell index. The
        # continuous spatial losses evaluate these directly rather than
        # the cell centers, so sub-cell geometry is preserved.
        shot_xy_np = np.stack([x_coord[valid], y_coord[valid]], axis=1, dtype=np.float32)
        if len(df) == 0:
            raise ValueError("no in-court shots remained after filtering")

        # Latest anchor on or before each shot's date. Shots before the
        # first anchor (index -1) have no causal snapshot and are dropped.
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

        # Within-game shot-history features h_{n,r}, built from earlier
        # shots of the same game on the filtered frame so rows align
        # with cells / ctx / shot_xy.
        h_within_game_np = compute_within_game_features(df)
        if h_within_game_np.shape != (len(df), WITHIN_GAME_DIM):
            raise ValueError(
                f"compute_within_game_features produced {h_within_game_np.shape}; "
                f"expected ({len(df)}, {WITHIN_GAME_DIM})"
            )

        # Causal prior-outcome summary o_{n,r}. Zero-filled when there is
        # no ``made`` column, so a residual encoder with
        # ``outcome_dim > 0`` can be used without outcome data.
        if "made" in df.columns:
            prior_outcome_np = compute_prior_outcome_features(df)
        else:
            prior_outcome_np = np.zeros((len(df), PRIOR_OUTCOME_DIM), dtype=np.float32)

        # Optionally append the causal prior-shot KDE feature to o_{n,r};
        # the residual encoder's ``outcome_dim`` must then be
        # PRIOR_OUTCOME_DIM + PRIOR_SHOT_KDE_DIM.
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

        # Padded sequence of earlier same-game shots for the optional
        # within-game recurrent encoder. Always built, so the batch tuple
        # has a fixed layout; per-row memory is bounded by MAX_PRIOR_SHOTS.
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

        # Per-game integer index over (player_id, game_id) pairs, so each
        # player-game has its own count target. pd.factorize assigns
        # 0..n_games-1 in order of first appearance.
        #
        # The key separator must not be a null byte: pandas 3.x factorizes
        # object arrays with a C-string hash that truncates at the null,
        # which would merge all games of a player into one index. ``|``
        # never occurs in player or game IDs.
        game_keys = (df["player_id"].astype(str) + "|" + df["game_id"].astype(str)).to_numpy()
        game_id_np, _ = pd.factorize(game_keys, sort=False)
        game_id_np = game_id_np.astype(np.int64)
        n_games = int(game_id_np.max()) + 1
        k_obs_np = np.bincount(game_id_np, minlength=n_games).astype(np.int64)

        # Per-game pregame context: the context of the game's first shot.
        # The count head consumes only this pregame representative (the
        # count loss is per game). The first occurrence of each game is
        # where the stably sorted key changes.
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
        # PRIOR_OUTCOME_DIM, plus PRIOR_SHOT_KDE_DIM with the spatial-Hawkes
        # feature.
        self.outcome_feature_dim: int = int(prior_outcome_np.shape[-1])
        self.with_spatial_hawkes_residual: bool = bool(with_spatial_hawkes_residual)
        self.spatial_hawkes_sigma_ft: float = float(spatial_hawkes_sigma_ft)
        self.prior_seq: Tensor = torch.from_numpy(prior_seq_np).to(dtype=dtype)
        self.prior_lengths: Tensor = torch.from_numpy(prior_lengths_np)
        self.per_game: PerGameTable = PerGameTable(
            x_n_raw=torch.from_numpy(x_n_raw_game).to(dtype=dtype),
            k_obs=torch.from_numpy(k_obs_np),
        )
        # Per-shot metadata in dataset row order, so callers can align
        # external tables on (player_id, date, starter) without
        # replicating the filters above.
        meta_cols = ["player_id", "date"]
        if "starter" in df.columns:
            meta_cols.append("starter")
        self.per_shot_meta: pd.DataFrame = df[meta_cols].reset_index(drop=True)

    @property
    def n_shots(self) -> int:
        """Number of shots in the dataset."""
        return int(self.cell_idx.shape[0])

    @property
    def n_games(self) -> int:
        """Number of player-games in the per-game table."""
        return int(self.per_game.k_obs.shape[0])

    @property
    def has_opponents(self) -> bool:
        """Whether an opponent vocabulary was given (``opp_idx`` is meaningful)."""
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
        """Return a random sub-dataset of ``n_shots`` rows, seeded by ``seed``.

        Useful for quick training runs on a fraction of the data. The
        per-game table is rebuilt from the sampled rows, so ``k_obs``
        counts only the sampled shots of each game.

        Parameters
        ----------
        n_shots : int
            Number of shots to keep; must be positive. Values at or above
            :attr:`n_shots` return ``self`` unchanged.
        seed : int, default 0
            Seed of the sampling generator.

        Returns
        -------
        GibbsShotDataset
            The subset, sharing vocabularies and snapshot store with
            ``self``.

        Notes
        -----
        The subset does not carry :attr:`per_shot_meta`.
        """
        if n_shots <= 0:
            raise ValueError(f"n_shots must be > 0; got {n_shots}")
        if n_shots >= self.n_shots:
            return self

        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(self.n_shots, size=n_shots, replace=False)).astype(np.int64)
        keep_t = torch.from_numpy(keep)

        # Bypass __init__: the source tensors are already validated.
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

        # Remap the kept games to contiguous indices [0, n_kept_games).
        orig_game = self.game_idx[keep_t]
        kept_game_ids, remapped_game = torch.unique(orig_game, return_inverse=True)
        out.game_idx = remapped_game.to(torch.int64)
        n_kept_games = int(kept_game_ids.shape[0])

        # Per-game context: first sampled shot of each game.
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
