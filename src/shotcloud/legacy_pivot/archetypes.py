"""Archetype dictionary and mixture network for the offensive prior.

Deprecated; retained to reproduce the archetype-prior ablations of the
grid-cell decoder. Superseded by the pooled analogue-player support of
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

This module provides two complementary objects:

* :class:`ArchetypeDictionary` — a frozen, indexed registry of
  per-anchor archetype surfaces ``{A_k^{(t_i)}}_{i=1..T, k=1..K}``.
  Each surface is a probability distribution over court cells. The
  surfaces are NOT learnable parameters; they are computed once by
  the chronological pretraining pass
  (``scripts/legacy_pivot/pretrain_snapshots_legacy.py``) and frozen for the duration of
  joint training. This is what enforces the Causal Snapshot Principle
  for the archetype basis: the basis at time ``t`` was fit from data
  with date ``< t_{i(t)}``.

* :class:`ArchetypeMixture` — a small, learnable network producing
  ``rho_xi(p, x_n) in Delta^{K-1}`` via the semi-structured form
  ``g_{xi,k}(p, x_n) = b_k + a_k^T r_p + d_k^T x_n``. Each archetype's
  role-vector signature ``a_k in R^{ROLE_PROFILE_DIM}`` is a directly
  interpretable artifact of training, which is why the head is
  semi-structured rather than a black-box MLP.

Together they implement the archetype branch of the offensive prior:

    q^arch_xi(c | p, x_n) = sum_k rho_{xi,k}(p, x_n) * A_k^{(t_{i(t)})}(c)

Inference flow at training time::

    bundle = snapshot_store.get_snapshot(date)
    snapshot_idx = ...  # the index into ArchetypeDictionary's stacked tensor
    x_n = encoder.transform(df)                 # (B, CONTEXT_DIM)
    mixture = archetype_mixture(x_n)            # (B, K), softmax-normalized
    q_arch = archetype_dict(snapshot_idx, mixture)  # (B, C)
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from shotcloud.data.context import CONTEXT_DIM, FEATURE_LAYOUT
from shotcloud.data.snapshots import ROLE_PROFILE_DIM, SnapshotStore

#: Default archetype count ``K``.
DEFAULT_K: int = 8


class ArchetypeDictionary(nn.Module):
    """Frozen per-anchor archetype surfaces ``{A_k^{(t_i)}}``.

    Holds a single ``(T, K, C)`` tensor of probability distributions
    over court cells, registered as a non-trainable buffer. Indexed
    lookups by snapshot return the bundle of K surfaces valid for
    the corresponding anchor; batched mixtures combine them via
    learnable weights produced by :class:`ArchetypeMixture`.

    Attributes
    ----------
    surfaces : Tensor
        Shape ``(T, K, C)``, dtype ``float32``, registered as a
        buffer (not a parameter). Each ``surfaces[i, k, :]`` is a
        probability distribution: non-negative and sums to 1.
    n_anchors, n_archetypes, n_cells : int
        Convenient shape accessors.

    Notes
    -----
    The surfaces are *frozen*. They never receive gradient updates
    during training. This frozen-after-pretrain guarantee is what
    makes the basis time-causal: A_k^{(t_i)} was fit
    from shots with date ``< t_i`` during the chronological
    pretraining pass, and joint training cannot pollute it with
    later data.
    """

    surfaces: Tensor

    def __init__(self, surfaces: NDArray[np.float32] | Tensor) -> None:
        super().__init__()
        if isinstance(surfaces, np.ndarray):
            tensor = torch.from_numpy(surfaces.astype(np.float32, copy=False))
        else:
            tensor = surfaces.to(torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"surfaces must be 3-D (T, K, C); got shape {tuple(tensor.shape)}")
        if tensor.shape[0] == 0 or tensor.shape[1] == 0:
            raise ValueError(f"surfaces has empty dimension: shape {tuple(tensor.shape)}")
        if (tensor < 0).any():
            raise ValueError("surfaces must be non-negative")
        row_sums = tensor.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            max_dev = (row_sums - 1.0).abs().max().item()
            raise ValueError(f"surfaces rows must sum to 1 (max deviation {max_dev:.4g})")
        self.register_buffer("surfaces", tensor, persistent=True)

    @property
    def n_anchors(self) -> int:
        return int(self.surfaces.shape[0])

    @property
    def n_archetypes(self) -> int:
        return int(self.surfaces.shape[1])

    @property
    def n_cells(self) -> int:
        return int(self.surfaces.shape[2])

    @classmethod
    def from_snapshot_store(cls, store: SnapshotStore) -> ArchetypeDictionary:
        """Stack a :class:`SnapshotStore`'s archetype surfaces.

        Every bundle must have ``archetype_surfaces`` populated (i.e.
        the chronological pretrain pass has run). For testing without
        pretraining, use :meth:`uniform` instead.

        Raises
        ------
        ValueError
            If any bundle has ``archetype_surfaces is None`` or shapes
            are inconsistent across bundles.
        """
        surfaces_list: list[NDArray[np.float32]] = []
        for i, bundle in enumerate(store.bundles):
            if bundle.archetype_surfaces is None:
                raise ValueError(
                    f"bundle {i} (anchor {bundle.anchor_date}) has no "
                    f"archetype_surfaces; run scripts/pretrain_snapshots.py first"
                )
            surfaces_list.append(bundle.archetype_surfaces)
        shapes = {arr.shape for arr in surfaces_list}
        if len(shapes) != 1:
            raise ValueError(f"inconsistent archetype_surfaces shapes across bundles: {shapes}")
        stacked = np.stack(surfaces_list, axis=0)
        return cls(stacked)

    @classmethod
    def uniform(cls, n_anchors: int, n_archetypes: int, n_cells: int) -> ArchetypeDictionary:
        """Construct a dictionary with uniform-over-cells surfaces.

        Useful as a development / testing fallback when no pretrained
        snapshot store is available. The mixture network can be
        exercised end-to-end against this; the resulting q_arch is
        always uniform, but other architectural pieces still work.
        """
        if n_anchors <= 0 or n_archetypes <= 0 or n_cells <= 0:
            raise ValueError(
                f"all dimensions must be positive; got "
                f"n_anchors={n_anchors}, n_archetypes={n_archetypes}, n_cells={n_cells}"
            )
        surfaces = np.full((n_anchors, n_archetypes, n_cells), 1.0 / n_cells, dtype=np.float32)
        return cls(surfaces)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def density(self, snapshot_idx: int, k: int | None = None) -> Tensor:
        """Return one snapshot's archetype surfaces.

        Parameters
        ----------
        snapshot_idx : int
            Index into the store's chronological bundle list.
        k : int, optional
            Specific archetype index. If None, return all K surfaces
            at this snapshot, shape ``(K, C)``. If given, return the
            single surface, shape ``(C,)``.
        """
        if not (0 <= snapshot_idx < self.n_anchors):
            raise IndexError(f"snapshot_idx {snapshot_idx} out of range [0, {self.n_anchors})")
        if k is None:
            return self.surfaces[snapshot_idx]
        if not (0 <= k < self.n_archetypes):
            raise IndexError(f"k {k} out of range [0, {self.n_archetypes})")
        return self.surfaces[snapshot_idx, k]

    def forward(self, snapshot_idx: Tensor, mixture: Tensor) -> Tensor:
        """Compute the archetype prior density ``q^arch(c | p, x_n)`` per row.

        Parameters
        ----------
        snapshot_idx : Tensor
            Shape ``(B,)``, dtype int64. Index into the store's
            stacked surfaces for each batch row. Values must be in
            ``[0, n_anchors)``.
        mixture : Tensor
            Shape ``(B, K)``, dtype float. Per-row archetype mixture
            weights produced by :class:`ArchetypeMixture`. Each row
            must sum to 1 (not validated for speed; the
            :class:`ArchetypeMixture` softmax already guarantees it).

        Returns
        -------
        Tensor
            Shape ``(B, n_cells)``: ``q^arch[b, c] = sum_k mixture[b, k] *
            surfaces[snapshot_idx[b], k, c]``. Each row is a
            probability distribution over cells (sums to 1) by
            construction.
        """
        if snapshot_idx.dim() != 1:
            raise ValueError(f"snapshot_idx must be 1-D; got shape {tuple(snapshot_idx.shape)}")
        if mixture.dim() != 2 or mixture.shape[1] != self.n_archetypes:
            raise ValueError(
                f"mixture must have shape (B, K) with K={self.n_archetypes}; "
                f"got {tuple(mixture.shape)}"
            )
        if mixture.shape[0] != snapshot_idx.shape[0]:
            raise ValueError(
                f"snapshot_idx and mixture batch dims must match; "
                f"got {snapshot_idx.shape[0]} vs {mixture.shape[0]}"
            )
        # surfaces[snapshot_idx]: (B, K, C); einsum over k.
        surfaces_per_row = self.surfaces[snapshot_idx]
        return torch.einsum("bk,bkc->bc", mixture, surfaces_per_row)


class ArchetypeMixture(nn.Module):
    """Semi-structured per-player-context archetype mixture network.

    Implements

    .. math::
        \\rho_{\\xi,k}(p, x_n) = \\mathrm{softmax}_k\\bigl[
            b_k + a_k^\\top r_p + d_k^\\top x_n
        \\bigr]

    where ``r_p`` is the player's role profile (the
    ``role_profile`` slice of ``x_n``) and ``x_n`` is the canonical
    per-shot context vector.

    Parameters
    ----------
    n_archetypes : int, default :data:`DEFAULT_K`
        Number of archetypes ``K``.
    role_profile_dim : int, default :data:`ROLE_PROFILE_DIM`
        Width of ``r_p``. Must match the slice
        ``FEATURE_LAYOUT["role_profile"]``.
    context_dim : int, default :data:`CONTEXT_DIM`
        Width of ``x_n``. Must match the encoder.

    Attributes
    ----------
    bias : nn.Parameter
        Shape ``(K,)``. Per-archetype bias ``b_k``.
    role_weight : nn.Parameter
        Shape ``(K, role_profile_dim)``. Per-archetype role-vector
        signature ``a_k``. Read off the trained model for archetype
        interpretation: each row tells you which role coordinates
        the archetype emphasizes.
    context_weight : nn.Parameter
        Shape ``(K, context_dim)``. Per-archetype context coefficient
        ``d_k``. Includes the role slice; it overlaps with `role_weight`
        in the role coordinates by design.

    Notes
    -----
    This is a tiny network: total parameters are
    ``K * (1 + role_profile_dim + context_dim) = 288`` at
    ``K=8, ROLE_PROFILE_DIM=8, CONTEXT_DIM=27``. As a globally pooled
    parameter it is fit on the whole training window rather than
    walk-forward; its small size bounds the leakage this
    parameter sharing can introduce.
    """

    def __init__(
        self,
        n_archetypes: int = DEFAULT_K,
        role_profile_dim: int = ROLE_PROFILE_DIM,
        context_dim: int = CONTEXT_DIM,
    ) -> None:
        super().__init__()
        if n_archetypes <= 0:
            raise ValueError(f"n_archetypes must be positive, got {n_archetypes}")
        if role_profile_dim <= 0:
            raise ValueError(f"role_profile_dim must be positive, got {role_profile_dim}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")

        self.n_archetypes = n_archetypes
        self.role_profile_dim = role_profile_dim
        self.context_dim = context_dim

        # Cache the role-profile slice into x_n as ints so forward is
        # tensor-only (avoids the slice lookup per batch).
        rp_slice = FEATURE_LAYOUT["role_profile"]
        self._role_start = int(rp_slice.start)
        self._role_stop = int(rp_slice.stop)
        if self._role_stop - self._role_start != role_profile_dim:
            raise ValueError(
                f"FEATURE_LAYOUT['role_profile'] has width "
                f"{self._role_stop - self._role_start}, expected {role_profile_dim}"
            )

        self.bias = nn.Parameter(torch.zeros(n_archetypes))
        self.role_weight = nn.Parameter(torch.zeros(n_archetypes, role_profile_dim))
        self.context_weight = nn.Parameter(torch.zeros(n_archetypes, context_dim))

    def forward(self, x_n: Tensor, r_p: Tensor | None = None) -> Tensor:
        """Compute mixture weights ``rho_xi(p, x_n)``.

        Parameters
        ----------
        x_n : Tensor
            Shape ``(B, context_dim)``. The per-shot context vector
            consumed by the per-archetype context coefficient
            ``d_k^T x_n``. When the caller has not applied an
            :class:`~shotcloud.models.ContextMLP`, this is the raw
            ``\\tilde x_n`` produced by
            :class:`shotcloud.data.ContextEncoder` --- its named
            slices are intact and ``r_p`` can be left as ``None``.
            When the caller has run ``x_n = f_ctx(\\tilde x_n)``, the
            named-slice layout no longer holds and the role profile
            must be supplied separately via ``r_p``.
        r_p : Tensor, optional
            Shape ``(B, role_profile_dim)``. Per-shot role profile
            ``r_p`` produced by
            :class:`shotcloud.data.SnapshotStore`. When ``None``
            (default), it is pulled from ``x_n[..., role_profile]``
            using the :data:`shotcloud.data.context.FEATURE_LAYOUT`
            slice convention. Pass explicitly when ``x_n`` has been
            transformed by an MLP that does not preserve that layout.

        Returns
        -------
        Tensor
            Shape ``(B, n_archetypes)``. Each row is a probability
            distribution over the K archetypes (rows sum to 1).
        """
        if x_n.dim() != 2:
            raise ValueError(f"x_n must be (B, context_dim); got {tuple(x_n.shape)}")
        if x_n.shape[-1] != self.context_dim:
            raise ValueError(f"x_n has last dim {x_n.shape[-1]}, expected {self.context_dim}")
        if r_p is None:
            r_p = x_n[..., self._role_start : self._role_stop]
        else:
            if r_p.dim() != 2:
                raise ValueError(f"r_p must be (B, role_profile_dim); got {tuple(r_p.shape)}")
            if r_p.shape[-1] != self.role_profile_dim:
                raise ValueError(
                    f"r_p has last dim {r_p.shape[-1]}, expected {self.role_profile_dim}"
                )
            if r_p.shape[0] != x_n.shape[0]:
                raise ValueError(f"r_p batch dim {r_p.shape[0]} does not match x_n {x_n.shape[0]}")
        # logits: (B, K) = bias + r_p @ a^T + x_n @ d^T
        logits = self.bias + r_p @ self.role_weight.t() + x_n @ self.context_weight.t()
        return torch.softmax(logits, dim=-1)

    def initialize_from_pretrained(
        self,
        bias: NDArray[np.float32] | Tensor,
        role_weight: NDArray[np.float32] | Tensor,
        context_weight: NDArray[np.float32] | Tensor,
    ) -> None:
        """Populate parameters from a pretrained closed-form fit.

        After the snapshot pretraining pass, the archetype mixture can be
        initialized by ridge regression of
        ``log rho^{(t_T)}_{p, k}`` on the role profile and pregame
        context. This method copies those fitted coefficients in
        place. Sizes must match the constructor arguments.
        """
        with torch.no_grad():
            self.bias.copy_(_to_tensor(bias).reshape(self.n_archetypes))
            self.role_weight.copy_(
                _to_tensor(role_weight).reshape(self.n_archetypes, self.role_profile_dim)
            )
            self.context_weight.copy_(
                _to_tensor(context_weight).reshape(self.n_archetypes, self.context_dim)
            )


def _to_tensor(x: NDArray[np.float32] | Tensor) -> Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32, copy=False))
    return x.to(torch.float32)


__all__ = [
    "DEFAULT_K",
    "ArchetypeDictionary",
    "ArchetypeMixture",
]
