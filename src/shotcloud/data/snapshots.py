r"""Snapshot store: causal registry of features derived from training data.

A :class:`SnapshotStore` holds a sequence of :class:`SnapshotBundle`
objects, one per anchor date (typically monthly). Each bundle carries
the slow-moving features valid at its anchor: per-player role profiles
and position mixtures, per-opponent efficiency bins, per-player and
per-opponent indices of prior shots, and, optionally, archetype
surfaces and mixtures for the legacy grid-cell decoder. Consumers read
these features only through::

    bundle = snapshot_store.get_snapshot(game_date)

which returns the bundle with the largest anchor ``t_i <= game_date``.

**Causality contract.** Every value in a bundle is computed from shots
dated strictly before the bundle's anchor, so a feature read for a shot
on date :math:`t` depends only on the information filtration
:math:`\mathcal F_{<t}`:

.. math::

    q(c \mid x_t) = q(c \mid \mathcal F_{<t}, x_t).

:meth:`SnapshotBundle.assert_causal` verifies the contract for a
bundle's indexed shot pools; the remaining fields are computed from the
same date-filtered sub-frame by :func:`build_snapshot_store_from_shots`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: Width of the per-player role profile vector r_p:
#: (rim_rate, paint_rate, midrange_rate, corner3_rate, atb3_rate,
#:  mean_dist, std_dist, shot_entropy). Computed by
#: :func:`shotcloud.data.role_profile.build_role_profiles`.
ROLE_PROFILE_DIM: Final[int] = 8

#: Width of the per-player soft position mixture pi_p^pos, a
#: distribution over (guard, wing, big).
POSITION_MIXTURE_DIM: Final[int] = 3

#: Number of opponent-efficiency buckets: quartiles of the FG% each
#: opponent allowed on shots before the snapshot anchor. This is an
#: *efficiency* summary, not a geometric one; the spatial effect of the
#: opponent is modeled separately by opponent reweighting.
N_OPP_EFFICIENCY_BINS: Final[int] = 4


@dataclass(frozen=True)
class SnapshotBundle:
    """Causal feature bundle valid at exactly one anchor date.

    Every field satisfies the F_<t invariant: each value is computed
    only from shots dated strictly before :attr:`anchor_date`. The
    model reads training-derived features through this interface at
    both training and inference time.

    Attributes
    ----------
    anchor_date : np.datetime64
        The anchor time t_i. The bundle is valid for predictions
        with shot date in `[anchor_date, next_anchor_date)`.
    player_ids : NDArray[np.int64]
        Sorted array of player IDs active at this anchor (i.e.,
        with at least one shot in the causal window). The arrays
        :attr:`role_profiles` and :attr:`position_mixtures` are
        row-aligned with this index.
    role_profiles : NDArray[np.float32]
        Per-player role vectors r_p^(t_i), shape
        ``(n_players, ROLE_PROFILE_DIM)``. Order:
        (rim_rate, paint_rate, midrange_rate, corner3_rate,
        atb3_rate, mean_dist, std_dist, shot_entropy). Computed
        causally by :func:`shotcloud.data.role_profile.build_role_profiles`.
    position_mixtures : NDArray[np.float32]
        Per-player soft position assignments pi_p^pos,(t_i) in the
        2-simplex, shape ``(n_players, POSITION_MIXTURE_DIM)`` over
        (guard, wing, big). Rows sum to 1.
    opp_codes : NDArray[np.str_]
        Sorted array of opponent codes (e.g., team abbreviations)
        active at this anchor.
    opp_efficiency_bins : NDArray[np.int8]
        Per-opponent efficiency bucket index in
        ``{0, 1, ..., N_OPP_EFFICIENCY_BINS-1}`` (0 = lowest FG%
        allowed); aligned with :attr:`opp_codes`.
    player_history_index : dict[int, NDArray[np.int64]]
        Per-player int64 indices into the source shot table. All
        indices point to shots with date strictly before
        :attr:`anchor_date`. Players absent from the dict had no
        qualifying shots at this anchor.
    defensive_history_index : dict[str, NDArray[np.int64]]
        Per-opponent int64 indices into the source shot table for
        opp-allowed shots. Same causal invariant as
        :attr:`player_history_index`.
    archetype_surfaces : NDArray[np.float32] | None
        Archetype basis ``A_k^(t_i)``, shape ``(K, n_cells)``, each
        row a probability distribution over court cells. None when
        the bundle is built without an archetype fit (the default of
        ``scripts/pretrain_snapshots.py``).
    archetype_mixtures : NDArray[np.float32] | None
        Per-player archetype mixture weights ``rho_p^(t_i)``, shape
        ``(P, K)``, each row a simplex over the K archetypes. The
        row at index ``i`` corresponds to player ``player_ids[i]``.
        Used to initialize the learned head ``rho_xi(p, x_n)`` of
        :class:`shotcloud.legacy_pivot.archetypes.ArchetypeMixture`.
        None unless the archetype fit returns mixtures for this
        anchor.
    archetype_player_ids : NDArray[np.int64] | None
        Player IDs aligned with ``archetype_mixtures`` rows. May
        differ from :attr:`player_ids` if some players were
        excluded from the archetype fit (e.g., below ``--min-shots``).
    """

    anchor_date: np.datetime64
    player_ids: NDArray[np.int64]
    role_profiles: NDArray[np.float32]
    position_mixtures: NDArray[np.float32]
    opp_codes: NDArray[np.str_]
    opp_efficiency_bins: NDArray[np.int8]
    player_history_index: dict[int, NDArray[np.int64]]
    defensive_history_index: dict[str, NDArray[np.int64]]
    archetype_surfaces: NDArray[np.float32] | None = None
    archetype_mixtures: NDArray[np.float32] | None = None
    archetype_player_ids: NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        n_players = len(self.player_ids)
        n_opps = len(self.opp_codes)

        if self.role_profiles.shape != (n_players, ROLE_PROFILE_DIM):
            raise ValueError(
                f"role_profiles has shape {self.role_profiles.shape}, "
                f"expected ({n_players}, {ROLE_PROFILE_DIM})"
            )
        if self.position_mixtures.shape != (n_players, POSITION_MIXTURE_DIM):
            raise ValueError(
                f"position_mixtures has shape {self.position_mixtures.shape}, "
                f"expected ({n_players}, {POSITION_MIXTURE_DIM})"
            )
        if self.opp_efficiency_bins.shape != (n_opps,):
            raise ValueError(
                f"opp_efficiency_bins has shape {self.opp_efficiency_bins.shape}, "
                f"expected ({n_opps},)"
            )
        if n_players > 0:
            if (self.position_mixtures < 0).any():
                raise ValueError("position_mixtures must be non-negative")
            row_sums = self.position_mixtures.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                raise ValueError(
                    f"position_mixtures rows must sum to 1 (max deviation "
                    f"{abs(row_sums - 1.0).max():.4g})"
                )
        if self.archetype_surfaces is not None:
            if self.archetype_surfaces.ndim != 2:
                raise ValueError(
                    f"archetype_surfaces must be 2-D (K, n_cells); got shape "
                    f"{self.archetype_surfaces.shape}"
                )
            if (self.archetype_surfaces < 0).any():
                raise ValueError("archetype_surfaces must be non-negative")
            row_sums = self.archetype_surfaces.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-4):
                raise ValueError(
                    f"archetype_surfaces rows must sum to 1 (max deviation "
                    f"{abs(row_sums - 1.0).max():.4g})"
                )
        if self.archetype_mixtures is not None:
            if self.archetype_surfaces is None:
                raise ValueError("archetype_mixtures requires archetype_surfaces to be set")
            if self.archetype_player_ids is None:
                raise ValueError("archetype_mixtures requires archetype_player_ids to be set")
            # archetype_player_ids: 1-D int with strictly increasing entries
            # so downstream binary-search lookups (analogous to player_idx)
            # work correctly.
            if self.archetype_player_ids.ndim != 1:
                raise ValueError(
                    f"archetype_player_ids must be 1-D; got shape {self.archetype_player_ids.shape}"
                )
            if not np.issubdtype(self.archetype_player_ids.dtype, np.integer):
                raise ValueError(
                    f"archetype_player_ids must be integer dtype; got "
                    f"{self.archetype_player_ids.dtype}"
                )
            if len(self.archetype_player_ids) > 1 and not np.all(
                np.diff(self.archetype_player_ids) > 0
            ):
                raise ValueError(
                    "archetype_player_ids must be strictly increasing (unique + sorted)"
                )
            K = int(self.archetype_surfaces.shape[0])
            P_fit = int(self.archetype_player_ids.shape[0])
            if self.archetype_mixtures.shape != (P_fit, K):
                raise ValueError(
                    f"archetype_mixtures has shape {self.archetype_mixtures.shape}, "
                    f"expected ({P_fit}, {K})"
                )
            if (self.archetype_mixtures < 0).any():
                raise ValueError("archetype_mixtures must be non-negative")
            if P_fit > 0:
                row_sums = self.archetype_mixtures.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-4):
                    raise ValueError(
                        f"archetype_mixtures rows must sum to 1 (max deviation "
                        f"{abs(row_sums - 1.0).max():.4g})"
                    )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def player_idx(self, player_id: int) -> int | None:
        """Return the row index of `player_id` in this bundle, or None if absent."""
        idx = int(np.searchsorted(self.player_ids, int(player_id)))
        if idx < len(self.player_ids) and int(self.player_ids[idx]) == int(player_id):
            return idx
        return None

    def opp_idx(self, opp: str) -> int | None:
        """Return the row index of ``opp`` in this bundle, or None if absent."""
        idx = int(np.searchsorted(self.opp_codes, str(opp)))
        if idx < len(self.opp_codes) and str(self.opp_codes[idx]) == str(opp):
            return idx
        return None

    def role_profile(self, player_id: int) -> NDArray[np.float32] | None:
        """Return the player's role profile; None if not active at this anchor."""
        i = self.player_idx(player_id)
        if i is None:
            return None
        return cast("NDArray[np.float32]", self.role_profiles[i])

    def position_mixture(self, player_id: int) -> NDArray[np.float32] | None:
        """Return the player's position mixture; None if not active at this anchor."""
        i = self.player_idx(player_id)
        if i is None:
            return None
        return cast("NDArray[np.float32]", self.position_mixtures[i])

    def opp_strength_bin(self, opp: str) -> int | None:
        """Return the opponent's efficiency bin; None if not seen at this anchor."""
        i = self.opp_idx(opp)
        if i is None:
            return None
        return int(self.opp_efficiency_bins[i])

    def history_for(self, player_id: int) -> NDArray[np.int64]:
        """Return the player's causal shot indices (empty if none)."""
        return self.player_history_index.get(int(player_id), np.array([], dtype=np.int64))

    def defensive_history_for(self, opp: str) -> NDArray[np.int64]:
        """Return the shot indices allowed by ``opp`` (empty if none)."""
        return self.defensive_history_index.get(str(opp), np.array([], dtype=np.int64))

    # ------------------------------------------------------------------
    # Causality verification
    # ------------------------------------------------------------------

    def assert_causal(self, shots: pd.DataFrame) -> None:
        """Raise AssertionError if any indexed shot has date >= anchor_date.

        Every shot index referenced by :attr:`player_history_index` and
        :attr:`defensive_history_index` must point to a row of
        ``shots`` dated strictly before :attr:`anchor_date`.

        Parameters
        ----------
        shots : DataFrame
            The source shot table that the bundle indexes into. Must
            contain a ``date`` column convertible to numpy datetime64.

        Raises
        ------
        ValueError
            If ``shots`` has no ``date`` column.
        AssertionError
            With a per-violation message listing the offending
            player or opponent and the leak count.
        """
        if "date" not in shots.columns:
            raise ValueError("shots frame must have a 'date' column")
        all_dates = pd.to_datetime(shots["date"]).to_numpy(dtype="datetime64[D]")
        anchor = self.anchor_date.astype("datetime64[D]")

        for pid, indices in self.player_history_index.items():
            if len(indices) == 0:
                continue
            leak = (all_dates[indices] >= anchor).sum()
            if leak > 0:
                raise AssertionError(
                    f"player {pid}: {int(leak)} of {len(indices)} indexed shots "
                    f"have date >= anchor {anchor}"
                )

        for opp, indices in self.defensive_history_index.items():
            if len(indices) == 0:
                continue
            leak = (all_dates[indices] >= anchor).sum()
            if leak > 0:
                raise AssertionError(
                    f"opp {opp}: {int(leak)} of {len(indices)} indexed shots "
                    f"have date >= anchor {anchor}"
                )


@dataclass(frozen=True)
class SnapshotStore:
    """Indexed registry of :class:`SnapshotBundle` objects.

    The store implements the time-causal lookup
    ``S(t) = bundle at i(t), i(t) = max{i : t_i <= t}`` via binary
    search on anchor dates, and is the single object through which the
    model reads causal, training-data-derived features.

    Bundles must be passed in chronological order. The store
    enforces strict monotonicity of anchor dates at construction.

    Parameters
    ----------
    bundles : tuple[SnapshotBundle, ...]
        Sorted ascending by `anchor_date`. Empty stores are not
        permitted.
    """

    bundles: tuple[SnapshotBundle, ...]

    def __post_init__(self) -> None:
        if len(self.bundles) == 0:
            raise ValueError("SnapshotStore requires at least one bundle")
        anchors = [b.anchor_date.astype("datetime64[D]") for b in self.bundles]
        for i in range(len(anchors) - 1):
            if anchors[i + 1] <= anchors[i]:
                raise ValueError(
                    f"anchor_dates must be strictly increasing; "
                    f"got {anchors[i]} >= {anchors[i + 1]} at position {i}"
                )

    def __len__(self) -> int:
        """Number of bundles."""
        return len(self.bundles)

    @property
    def anchor_dates(self) -> NDArray[np.datetime64]:
        """All anchor dates as a sorted datetime64[D] array."""
        return cast(
            "NDArray[np.datetime64]",
            np.array(
                [b.anchor_date.astype("datetime64[D]") for b in self.bundles],
                dtype="datetime64[D]",
            ),
        )

    def get_snapshot_index(self, game_date: np.datetime64 | pd.Timestamp | str) -> int:
        """Return the index `i(t) = max{i : t_i <= game_date}`.

        Raises
        ------
        ValueError
            If `game_date` precedes the earliest anchor (no causal
            snapshot is available for that date).
        """
        date = _to_date(game_date)
        first_anchor = self.bundles[0].anchor_date.astype("datetime64[D]")
        if date < first_anchor:
            raise ValueError(
                f"game_date {date} precedes the earliest snapshot anchor "
                f"{first_anchor}; no causal snapshot is available. Either "
                f"filter shots to date >= {first_anchor} before training, "
                f"or extend the anchor grid further back in time."
            )
        idx: int = int(np.searchsorted(self.anchor_dates, date, side="right")) - 1
        return idx

    def get_snapshot(self, game_date: np.datetime64 | pd.Timestamp | str) -> SnapshotBundle:
        """Return the snapshot bundle valid for predictions at `game_date`."""
        return self.bundles[self.get_snapshot_index(game_date)]

    def assert_causal(self, shots: pd.DataFrame) -> None:
        """Run :meth:`SnapshotBundle.assert_causal` on every bundle.

        Linear in the total number of indexed shots across bundles;
        call it before persisting a store.
        """
        for bundle in self.bundles:
            bundle.assert_causal(shots)


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------


#: Role-profile hook: maps a causal shot sub-frame to ``{player_id: r_p}``.
RoleProfileFn = Callable[[pd.DataFrame], dict[int, NDArray[np.float32]]]
#: Position-mixture hook: maps a causal shot sub-frame to
#: ``{player_id: pi_p^pos}``.
PositionMixtureFn = Callable[[pd.DataFrame], dict[int, NDArray[np.float32]]]
_ArchetypeTriple = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]
#: Archetype fit takes ``(filtered_shots, anchor_date)`` and returns either:
#: * ``None`` (no archetype fit at this anchor),
#: * surfaces ``A`` of shape ``(K, n_cells)`` (surfaces only), or
#: * a triple ``(A, rho, player_ids)`` carrying surfaces, per-player
#:   mixture weights of shape ``(P, K)``, and the int64 player IDs that
#:   align with ``rho``'s rows, so the bundle can persist mixtures for
#:   initializing :class:`shotcloud.legacy_pivot.archetypes.ArchetypeMixture`.
#:
#: ``anchor_date`` lets the callable persist per-anchor checkpoints
#: (``scripts/pretrain_snapshots.py --checkpoint-dir``); callables that
#: do not checkpoint may ignore it.
ArchetypeFitFn = Callable[
    [pd.DataFrame, np.datetime64],
    "NDArray[np.float32] | _ArchetypeTriple | None",
]


def _to_date(d: np.datetime64 | pd.Timestamp | str) -> np.datetime64:
    """Coerce a date-like to ``datetime64[D]``."""
    if isinstance(d, np.datetime64):
        return cast("np.datetime64", d.astype("datetime64[D]"))
    if isinstance(d, pd.Timestamp):
        return cast("np.datetime64", d.to_datetime64().astype("datetime64[D]"))
    arr: Any = np.datetime64(d)
    return cast("np.datetime64", arr.astype("datetime64[D]"))


def build_snapshot_store_from_shots(
    shots: pd.DataFrame,
    anchor_dates: list[np.datetime64] | NDArray[np.datetime64],
    *,
    role_profile_fn: RoleProfileFn | None = None,
    position_mixture_fn: PositionMixtureFn | None = None,
    archetype_fit_fn: ArchetypeFitFn | None = None,
) -> SnapshotStore:
    """Construct a :class:`SnapshotStore` by chronological pass over shots.

    For each anchor `t_i`, this builder:

    1. Filters `shots` to rows with ``date < t_i``.
    2. Computes per-player and per-opponent shot indices into the
       original shot table (the causal pools).
    3. Assigns each opponent an efficiency bin from the quartiles of
       FG% allowed over the filtered shots (all zeros when fewer than
       ``N_OPP_EFFICIENCY_BINS`` opponents are present).
    4. Optionally calls ``role_profile_fn(filtered_shots)`` for role
       profiles and ``position_mixture_fn(filtered_shots)`` for soft
       position assignments (renormalized to the simplex; invalid rows
       become uniform). Without a hook, role profiles are zero and
       position mixtures uniform.
    5. Optionally calls ``archetype_fit_fn(filtered_shots, anchor_date)``
       for archetype surfaces (and mixtures). Without it,
       :attr:`SnapshotBundle.archetype_surfaces` is None.
    6. Packs everything into a frozen :class:`SnapshotBundle`.

    The hooks keep this module independent of the profile and
    archetype implementations; tests can pass simple stubs.

    Parameters
    ----------
    shots : pd.DataFrame
        Output of :func:`shotcloud.data.load_shots` or equivalent.
        Must have columns ``player_id`` and ``date``. ``opponent`` is
        needed for the defensive pools and ``opponent`` plus ``made``
        for the efficiency bins; without them those fields are empty or
        zero.
    anchor_dates : sequence of datetime64
        Strictly-increasing monthly anchors (or other cadence). The
        store can only answer queries for dates `>= anchor_dates[0]`.
    role_profile_fn, position_mixture_fn, archetype_fit_fn : callable, optional
        Computation hooks. Each receives the filtered (causal)
        sub-frame for the current anchor; ``archetype_fit_fn`` also
        receives the anchor date.

    Returns
    -------
    SnapshotStore
        Anchors with no prior shots are skipped.

    Raises
    ------
    ValueError
        If ``shots`` is empty or lacks ``date`` or ``player_id``, if
        ``anchor_dates`` is empty or not strictly increasing, or if no
        anchor has any prior shots.
    """
    if len(shots) == 0:
        raise ValueError("empty shots frame")
    if "date" not in shots.columns:
        raise ValueError("shots frame must have a 'date' column")
    if "player_id" not in shots.columns:
        raise ValueError("shots frame must have a 'player_id' column")

    anchors_arr = np.asarray([_to_date(a) for a in anchor_dates], dtype="datetime64[D]")
    if len(anchors_arr) == 0:
        raise ValueError("anchor_dates must be non-empty")
    for i in range(len(anchors_arr) - 1):
        if anchors_arr[i + 1] <= anchors_arr[i]:
            raise ValueError(
                f"anchor_dates must be strictly increasing; "
                f"got {anchors_arr[i]} >= {anchors_arr[i + 1]} at position {i}"
            )

    # Pre-sort shots by date and keep original indices for the history pools.
    shots_sorted = shots.copy()
    shots_sorted["_orig_idx"] = np.arange(len(shots_sorted), dtype=np.int64)
    shots_sorted = shots_sorted.sort_values("date", kind="stable").reset_index(drop=True)
    sorted_dates = pd.to_datetime(shots_sorted["date"]).to_numpy(dtype="datetime64[D]")

    has_opp = "opponent" in shots_sorted.columns
    has_made = "made" in shots_sorted.columns

    bundles: list[SnapshotBundle] = []
    for t_i in anchors_arr:
        # Strict-causal filter: rows with date < t_i (date == t_i is excluded).
        cutoff = int(np.searchsorted(sorted_dates, t_i, side="left"))
        if cutoff == 0:
            continue
        sub = shots_sorted.iloc[:cutoff]

        # ---- Per-player history index ----
        player_history_index: dict[int, NDArray[np.int64]] = {}
        for pid, group in sub.groupby("player_id", sort=True):
            player_history_index[int(pid)] = group["_orig_idx"].to_numpy(dtype=np.int64)
        player_ids = np.array(sorted(player_history_index.keys()), dtype=np.int64)

        # ---- Per-opponent defensive history index ----
        defensive_history_index: dict[str, NDArray[np.int64]] = {}
        if has_opp:
            for opp, group in sub.groupby("opponent", sort=True):
                if pd.isna(opp):
                    continue
                defensive_history_index[str(opp)] = group["_orig_idx"].to_numpy(dtype=np.int64)
        opp_codes = np.array(sorted(defensive_history_index.keys()), dtype=np.str_)

        # ---- Role profiles ----
        role_profiles = np.zeros((len(player_ids), ROLE_PROFILE_DIM), dtype=np.float32)
        if role_profile_fn is not None:
            role_dict = role_profile_fn(sub)
            for i, pid in enumerate(player_ids):
                vec = role_dict.get(int(pid))
                if vec is not None:
                    role_profiles[i] = np.asarray(vec, dtype=np.float32)

        # ---- Position mixtures ----
        position_mixtures = np.full(
            (len(player_ids), POSITION_MIXTURE_DIM),
            1.0 / POSITION_MIXTURE_DIM,
            dtype=np.float32,
        )
        if position_mixture_fn is not None:
            pos_dict = position_mixture_fn(sub)
            for i, pid in enumerate(player_ids):
                vec = pos_dict.get(int(pid))
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.shape != (POSITION_MIXTURE_DIM,):
                    raise ValueError(
                        f"position_mixture_fn returned shape {arr.shape} "
                        f"for player {pid}; expected ({POSITION_MIXTURE_DIM},)"
                    )
                # Clamp negative entries and fall back to uniform when the
                # mass is zero or non-finite, so the bundle stays valid.
                arr = np.clip(arr, 0.0, None)
                s = float(arr.sum())
                if not np.isfinite(s) or s <= 1e-8:
                    position_mixtures[i] = np.full(
                        POSITION_MIXTURE_DIM,
                        1.0 / POSITION_MIXTURE_DIM,
                        dtype=np.float32,
                    )
                else:
                    position_mixtures[i] = arr / s

        # ---- Opp-efficiency bins ----
        opp_efficiency_bins = np.zeros(len(opp_codes), dtype=np.int8)
        if has_made and has_opp and len(opp_codes) >= N_OPP_EFFICIENCY_BINS:
            opp_made = sub.dropna(subset=["opponent", "made"]).groupby("opponent")["made"].mean()
            try:
                buckets = pd.qcut(
                    opp_made, q=N_OPP_EFFICIENCY_BINS, labels=False, duplicates="drop"
                )
            except ValueError:
                buckets = pd.Series(0, index=opp_made.index)
            bin_map: dict[str, int] = {
                str(o): int(b) if not pd.isna(b) else 0 for o, b in buckets.items()
            }
            opp_efficiency_bins = np.array(
                [bin_map.get(str(o), 0) for o in opp_codes], dtype=np.int8
            )

        # ---- Archetype surfaces (and mixtures, when supplied) ----
        archetype_surfaces: NDArray[np.float32] | None = None
        archetype_mixtures: NDArray[np.float32] | None = None
        archetype_player_ids: NDArray[np.int64] | None = None
        if archetype_fit_fn is not None:
            fit_out = archetype_fit_fn(sub, t_i)
            if fit_out is None:
                pass
            elif isinstance(fit_out, tuple):
                archetype_surfaces, archetype_mixtures, archetype_player_ids = fit_out
            else:
                archetype_surfaces = fit_out

        bundle = SnapshotBundle(
            anchor_date=t_i,
            player_ids=player_ids,
            role_profiles=role_profiles,
            position_mixtures=position_mixtures,
            opp_codes=opp_codes,
            opp_efficiency_bins=opp_efficiency_bins,
            player_history_index=player_history_index,
            defensive_history_index=defensive_history_index,
            archetype_surfaces=archetype_surfaces,
            archetype_mixtures=archetype_mixtures,
            archetype_player_ids=archetype_player_ids,
        )
        bundles.append(bundle)

    if not bundles:
        raise ValueError(
            f"no anchor in {len(anchors_arr)} provided dates had any prior shots; "
            f"check anchor_dates ({anchors_arr[0]}..{anchors_arr[-1]}) vs shot dates "
            f"({sorted_dates[0]}..{sorted_dates[-1]})"
        )

    return SnapshotStore(bundles=tuple(bundles))


__all__ = [
    "N_OPP_EFFICIENCY_BINS",
    "POSITION_MIXTURE_DIM",
    "ROLE_PROFILE_DIM",
    "ArchetypeFitFn",
    "PositionMixtureFn",
    "RoleProfileFn",
    "SnapshotBundle",
    "SnapshotStore",
    "build_snapshot_store_from_shots",
]
