"""Cell-free defensive feasibility field (PR-D1a, pure module).

Implements the multiplicative defensive reweighting term

.. math::

    D_\\Delta(s_m \\mid d, x_n, h_{n,r})
    = \\beta_D \\,\\log \\sum_{j \\in \\mathcal D_d^{<t_n}}
        \\alpha_{\\delta,j}(c_{d,n,r}) \\, K_h(s_m - s_j^{\\mathrm{allow}}),

where the attention is bilinear with a per-shot recency decay:

.. math::

    b_{\\delta,j}
    = q_\\Delta(c_{d,n,r})^\\top k_\\Delta(z_j^{\\mathrm{def}})
      - \\lambda_{\\delta,\\mathrm{age}}\\,\\Delta t_j,
    \\quad
    \\alpha_{\\delta,j}
    = \\operatorname{softmax}_j(b_{\\delta,j}).

The defensive query context is

.. math::

    c_{d,n,r} = [x_n,\\, h_{n,r},\\, e_d,\\, v_d(t_n)],

with :math:`v_d(t_n)` the PR-D0.5 defense-feature vector
(team / zone / reliability blocks; see
:mod:`shotcloud.features.defense_features`) and :math:`e_d` a small
learnable opponent-id embedding. The per-allowed-shot key context
:math:`z_j^{\\mathrm{def}}` is **location-derived** in this PR (the
allowed-shot coordinate, distance to rim, and 8-zone one-hot); the
opponent-level reliability / scheme signal is carried entirely by
the *query* side via the feature artifact.

Scope contract (PR-D1a, "pure module"):

* Inputs are pre-gathered tensors. The module does **not** consume
  the retrieval cache or the feature artifact directly — PR-D1b's
  adapter does that gather step.
* Output is the additive support-logit term :math:`D_\\Delta(s_m)`
  for every offensive support shot, shape ``(B, M_off)``. Downstream
  the :class:`ContinuousMixtureSpatial` wrapper adds this to the
  collaborative + residual support logits before the gated subset
  softmax (PR-D2a).
* :math:`\\beta_D` is the warm-init scaling scalar (default
  ``1e-3``). Hard zero is retained only as a unit-test setting for
  the exact-no-op invariant; near-zero warm init keeps gradient
  flow live into ``q_Δ`` / ``k_Δ`` / ``λ_{δ,age}`` from step 0.
* Chunked KDE evaluation along the ``M_off`` query axis is
  mandatory. The naive ``(B, M_off, M_def)`` distance tensor is
  too large for MPS training at ``B=256, M_off≈1500, M_def≈1000``;
  the chunked path processes ``query_chunk_size`` query points at
  a time and never materializes the full ``M_off × M_def`` tensor.

Cold-start opponents (``def_mask.any(dim=-1) == False`` for some
rows) contribute :math:`D_\\Delta = 0` exactly — they pass through
the support softmax without affecting the offensive weighting. The
implementation handles them via a safe-mode dummy on the masked
softmax then a final ``torch.where`` zeroing, so no NaN ever
propagates.

What's intentionally out of scope (per PR-D1a build approval):

* No lineup features ``v_d^lineup(n,r)`` — deferred to PR-D0.6 /
  PR-D1.5.
* No real-data adapter / gather — PR-D1b owns that.
* No wrapper integration with :class:`ContinuousMixtureSpatial` —
  PR-D2a.
* No trainer flags / manifest / eval reconstruction — PR-D3.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn

from shotcloud.data.zones import N_ZONES, zone_from_xy_vectorized

if TYPE_CHECKING:
    from shotcloud.features.defense_features import DefenseFeatures
    from shotcloud.models.defensive_retrieval_cache import DefensiveRetrievalCache

#: Default isotropic kernel bandwidth in feet. Shared with the
#: offensive KDE per PR-D0.5 decision; the c → d ablation can sweep
#: a separate bandwidth later if needed.
DEFAULT_DEFENSE_BANDWIDTH_FT: Final[float] = 1.5

#: Default β_D warm init. See module docstring for rationale.
DEFAULT_DEFENSE_BETA_INIT: Final[float] = 1e-3

#: Default attention chunk size along the M_off axis.
DEFAULT_DEFENSE_QUERY_CHUNK_SIZE: Final[int] = 128

#: Per-shot key-context dim: (x, y, distance_to_rim, 8-zone-one-hot)
#: = 11. The 8-zone count comes from :data:`shotcloud.data.zones.N_ZONES`.
_KEY_INPUT_DIM: Final[int] = 2 + 1 + N_ZONES


def _location_key_context(def_xy: Tensor) -> Tensor:
    """Build per-allowed-shot key context ``z_j^def`` from coordinates.

    Returns a ``(B, M_def, _KEY_INPUT_DIM)`` tensor containing
    ``(x, y, distance_to_rim, zone_onehot)`` for each allowed shot.
    The zone one-hot is computed on a NumPy detour because the zone
    classifier is a NumPy routine in :mod:`shotcloud.data.zones`;
    this function is called in the no-grad sense (the result feeds
    into ``k_Δ`` which is the learnable part).
    """
    b, m, _ = def_xy.shape
    flat = def_xy.detach().reshape(b * m, 2).cpu().numpy()
    zone_flat = zone_from_xy_vectorized(flat[:, 0], flat[:, 1])
    # Out-of-court (zone == -1) gets an all-zero one-hot. This shouldn't
    # happen in practice because the retrieval cache filters by zone
    # upstream, but it's defensive.
    valid = zone_flat >= 0
    zone_onehot = torch.zeros(b * m, N_ZONES, dtype=def_xy.dtype, device=def_xy.device)
    if valid.any():
        valid_idx = torch.from_numpy(valid).to(def_xy.device)
        zone_idx = torch.from_numpy(zone_flat).clamp_min(0).to(def_xy.device)
        zone_onehot[valid_idx] = torch.nn.functional.one_hot(
            zone_idx[valid_idx], num_classes=N_ZONES
        ).to(def_xy.dtype)
    zone_onehot = zone_onehot.view(b, m, N_ZONES)
    # Distance to rim (origin) in feet.
    dist_to_rim = def_xy.norm(dim=-1, keepdim=True)  # (B, M_def, 1)
    return torch.cat([def_xy, dist_to_rim, zone_onehot], dim=-1)


class ContinuousAdaptiveDefensiveField(nn.Module):
    """Pure cell-free defensive feasibility module (PR-D1a).

    Parameters
    ----------
    n_opponents : int
        Size of the opponent vocab; controls the embedding table.
    context_dim : int
        Dimension of ``x_n`` (the offensive context vector).
    within_game_dim : int
        Dimension of ``h_n`` (within-game history). Pass ``0`` if no
        within-game history is fed to this module (the ``forward``
        will accept ``h_n=None`` in that case).
    defense_feature_dim : int
        Dimension of the per-(opp, snap) defense-feature vector.
        Should equal
        :data:`shotcloud.features.defense_features.DEFENSE_FEATURE_DIM`
        when wiring against the canonical artifact.
    opp_embed_dim : int, default 16
        Dimension of the learned opponent-id embedding ``e_d``.
        Keep small; the causal feature vector should carry most of
        the opponent signal.
    query_hidden_dim, key_hidden_dim : int, default 64
        Hidden width of the bilinear ``q_Δ`` and ``k_Δ`` MLPs.
    proj_dim : int, default 32
        Output dim of both ``q_Δ`` and ``k_Δ`` (the bilinear dot-
        product dim).
    bandwidth : float, default :data:`DEFAULT_DEFENSE_BANDWIDTH_FT`
        Isotropic Gaussian kernel bandwidth in feet. Shared with the
        offensive KDE in v1.
    beta_init : float, default :data:`DEFAULT_DEFENSE_BETA_INIT`
        Warm-init for the scaling scalar ``β_D``.
    lambda_age_init : float, default 0.0
        Initial value for the per-shot recency-decay coefficient
        ``λ_{δ,age}`` inside the attention. Zero is the simplest
        starting point because the cache layer already applied a
        recency window; the attention's own decay then learns over
        training.
    query_chunk_size : int, default :data:`DEFAULT_DEFENSE_QUERY_CHUNK_SIZE`
        Number of query points (offensive support shots) processed
        per chunk. The forward never materializes the full
        ``(B, M_off, M_def)`` distance tensor; chunk along ``M_off``
        and concatenate.
    """

    # Type-hint registered tensor buffers.
    bandwidth: Tensor

    def __init__(
        self,
        *,
        n_opponents: int,
        context_dim: int,
        within_game_dim: int,
        defense_feature_dim: int,
        opp_embed_dim: int = 16,
        query_hidden_dim: int = 64,
        key_hidden_dim: int = 64,
        proj_dim: int = 32,
        bandwidth: float = DEFAULT_DEFENSE_BANDWIDTH_FT,
        beta_init: float = DEFAULT_DEFENSE_BETA_INIT,
        lambda_age_init: float = 0.0,
        query_chunk_size: int = DEFAULT_DEFENSE_QUERY_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if n_opponents <= 0:
            raise ValueError(f"n_opponents must be positive; got {n_opponents}")
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive; got {context_dim}")
        if within_game_dim < 0:
            raise ValueError(f"within_game_dim must be ≥ 0; got {within_game_dim}")
        if defense_feature_dim <= 0:
            raise ValueError(f"defense_feature_dim must be positive; got {defense_feature_dim}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive; got {bandwidth}")
        if query_chunk_size <= 0:
            raise ValueError(f"query_chunk_size must be positive; got {query_chunk_size}")

        self._n_opponents = int(n_opponents)
        self._context_dim = int(context_dim)
        self._within_game_dim = int(within_game_dim)
        self._defense_feature_dim = int(defense_feature_dim)
        self._opp_embed_dim = int(opp_embed_dim)
        self._proj_dim = int(proj_dim)
        self._query_chunk_size = int(query_chunk_size)

        # Bandwidth as a buffer so it moves with .to(device) and is
        # exposed for inspection.
        self.register_buffer(
            "bandwidth", torch.tensor(float(bandwidth), dtype=torch.float32), persistent=False
        )

        # Learnable opponent-id embedding e_d.
        self.opp_embedding = nn.Embedding(n_opponents, opp_embed_dim)
        nn.init.normal_(self.opp_embedding.weight, mean=0.0, std=0.1)

        # Query network q_Δ: c_{d,n,r} → R^{proj_dim}.
        # Input dim: context_dim + within_game_dim + opp_embed_dim + defense_feature_dim.
        q_in = context_dim + within_game_dim + opp_embed_dim + defense_feature_dim
        self.q_net = nn.Sequential(
            nn.Linear(q_in, query_hidden_dim),
            nn.GELU(),
            nn.Linear(query_hidden_dim, proj_dim),
        )

        # Key network k_Δ: z_j^def → R^{proj_dim}.
        self.k_net = nn.Sequential(
            nn.Linear(_KEY_INPUT_DIM, key_hidden_dim),
            nn.GELU(),
            nn.Linear(key_hidden_dim, proj_dim),
        )

        # Recency-decay coefficient λ_{δ,age}. Positive values penalize
        # older shots in the attention.
        self.lambda_age = nn.Parameter(torch.tensor(float(lambda_age_init), dtype=torch.float32))

        # Warm-init scaling scalar β_D.
        self.beta_D = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    @property
    def query_chunk_size(self) -> int:
        return self._query_chunk_size

    def _build_query(
        self,
        x_n: Tensor,
        h_n: Tensor | None,
        opp_idx: Tensor,
        def_features: Tensor,
    ) -> Tensor:
        """Assemble the per-row query vector and project it via q_Δ.

        Returns shape ``(B, proj_dim)``.
        """
        e_d = self.opp_embedding(opp_idx)  # (B, opp_embed_dim)
        parts: list[Tensor] = [x_n]
        if self._within_game_dim > 0:
            if h_n is None:
                raise ValueError(
                    f"within_game_dim={self._within_game_dim} > 0 → h_n is required; "
                    "pass the per-row within-game history tensor"
                )
            parts.append(h_n)
        parts.append(e_d)
        parts.append(def_features)
        c = torch.cat(parts, dim=-1)
        q: Tensor = self.q_net(c)
        return q

    def _attention_log_alpha(
        self,
        *,
        q: Tensor,  # (B, proj_dim)
        def_xy: Tensor,  # (B, M_def, 2)
        def_mask: Tensor,  # (B, M_def) bool
        def_age_days: Tensor,  # (B, M_def) float
    ) -> Tensor:
        """Per-shot ``log α_{δ,j}`` from the bilinear attention.

        Returns shape ``(B, M_def)``. Rows whose ``def_mask`` is
        entirely False produce a "safe" log-softmax over a dummy
        single-slot to avoid NaN; downstream the row's ``D`` is
        zeroed entirely, so the dummy never affects the output.
        """
        z_def = _location_key_context(def_xy)  # (B, M_def, _KEY_INPUT_DIM)
        k = self.k_net(z_def)  # (B, M_def, proj_dim)
        # Bilinear score: (B, proj_dim) · (B, M_def, proj_dim) → (B, M_def).
        score = (q.unsqueeze(1) * k).sum(dim=-1)
        # Recency decay (positive λ_age penalizes older shots).
        score = score - self.lambda_age * def_age_days
        # Safe mask: ensure each row has ≥1 True so log_softmax is finite.
        # Rows with empty def_mask get a dummy slot at index 0; their D is
        # zeroed at the end of forward().
        has_any = def_mask.any(dim=-1)
        safe_mask = def_mask.clone()
        if (~has_any).any():
            safe_mask[~has_any, 0] = True
        # Float-mask infill (-1e30 is finite enough to round to 0 in
        # softmax but won't overflow when subtracted).
        score = score.masked_fill(~safe_mask, -1e30)
        log_alpha: Tensor = torch.log_softmax(score, dim=-1)
        return log_alpha

    def forward(
        self,
        *,
        query_xy: Tensor,  # (B, M_off, 2)
        def_xy: Tensor,  # (B, M_def, 2)
        def_mask: Tensor,  # (B, M_def) bool
        def_age_days: Tensor,  # (B, M_def) float
        def_features: Tensor,  # (B, D_def)
        x_n: Tensor,  # (B, context_dim)
        h_n: Tensor | None,  # (B, within_game_dim) or None
        opp_idx: Tensor,  # (B,) int64
    ) -> Tensor:
        """Compute :math:`D_\\Delta(s_m)` for every offensive support
        shot in every batch row.

        Returns shape ``(B, M_off)``. The output is :math:`\\beta_D`-
        scaled, so step-0 magnitudes are intentionally small (default
        ``β_D = 1e-3``).
        """
        _b, m_off, _ = query_xy.shape
        # Per-row query projection (one MLP forward per row).
        q = self._build_query(x_n, h_n, opp_idx, def_features)  # (B, proj_dim)
        # Per-row log α over allowed shots.
        log_alpha = self._attention_log_alpha(
            q=q, def_xy=def_xy, def_mask=def_mask, def_age_days=def_age_days
        )  # (B, M_def)

        # Chunked logsumexp over M_def for each chunk of M_off query
        # points. Never materializes the full (B, M_off, M_def) tensor.
        inv_two_h2 = 1.0 / (2.0 * float(self.bandwidth) ** 2)
        out_chunks: list[Tensor] = []
        chunk = self._query_chunk_size
        for start in range(0, m_off, chunk):
            stop = min(start + chunk, m_off)
            q_chunk = query_xy[:, start:stop, :]  # (B, q_chunk, 2)
            # Squared distance: (B, q_chunk, 1, 2) - (B, 1, M_def, 2) → (B, q_chunk, M_def).
            d2 = ((q_chunk.unsqueeze(2) - def_xy.unsqueeze(1)) ** 2).sum(dim=-1)
            log_k = -d2 * inv_two_h2  # (B, q_chunk, M_def)
            # Combine with attention log α (broadcast over q_chunk).
            log_terms = log_alpha.unsqueeze(1) + log_k  # (B, q_chunk, M_def)
            log_a_chunk: Tensor = torch.logsumexp(log_terms, dim=-1)  # (B, q_chunk)
            out_chunks.append(log_a_chunk)
        log_a = torch.cat(out_chunks, dim=-1)  # (B, M_off)

        # Per-row mean-center. The unscaled log-KDE field
        #   log Σ_j α_j K_h(s − s_j)
        # carries a large negative offset that depends on how far the
        # nearest support shot is from the query — at NBA scale with
        # h=1.5 ft, |log_a| can reach hundreds. That offset is a
        # *per-row constant* and cancels exactly under the downstream
        # support softmax in :class:`ContinuousMixtureSpatial`, so
        # subtracting it has zero effect on the spatial likelihood
        # while bringing the warm-init magnitude of ``D_Δ`` into the
        # `β_D`-controlled regime the test_4 invariant assumes.
        log_a = log_a - log_a.mean(dim=-1, keepdim=True)

        # β_D scaling.
        d_out = self.beta_D * log_a  # (B, M_off)

        # Zero out cold-start rows (def_mask empty → field is identically 0).
        has_any = def_mask.any(dim=-1)
        d_out = torch.where(has_any.unsqueeze(-1), d_out, torch.zeros_like(d_out))

        # Final NaN safety: should never fire if the safe-mask trick
        # worked, but a no-op guarantee here makes downstream
        # diagnostics robust to numerical surprises.
        d_out = torch.nan_to_num(d_out, nan=0.0, posinf=0.0, neginf=0.0)
        return d_out

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"n_opponents={self._n_opponents}, context_dim={self._context_dim}, "
            f"within_game_dim={self._within_game_dim}, "
            f"defense_feature_dim={self._defense_feature_dim}, "
            f"proj_dim={self._proj_dim}, bandwidth={float(self.bandwidth):.2f}, "
            f"beta_init={float(self.beta_D):.1e}, "
            f"query_chunk_size={self._query_chunk_size}, n_params={n_params}"
        )


# Silence the unused-import warning while still re-exporting for
# downstream consumers that may want to introspect the math constant.
_ = math


# ---------------------------------------------------------------------------
# PR-D1b: real-data gather adapter
# ---------------------------------------------------------------------------


def gather_defense_inputs(
    *,
    opp_idx: Tensor,
    snapshot_idx: Tensor,
    cache: DefensiveRetrievalCache,
    features: DefenseFeatures,
) -> dict[str, Tensor]:
    """Assemble per-row defensive inputs from the cache + features
    artifact, ready to feed :meth:`ContinuousAdaptiveDefensiveField.forward`.

    Parameters
    ----------
    opp_idx, snapshot_idx : Tensor of shape ``(B,)`` int64
        Per-row opponent vocab index and snapshot index.
    cache : DefensiveRetrievalCache
        PR-D0 allowed-shot retrieval cache. Provides
        ``def_idx / def_mask / global_xy / global_dates`` and the
        anchor-date tuple from its config.
    features : DefenseFeatures
        PR-D0.5 per-(opp, snap) feature tensor.

    Returns
    -------
    dict
        Keys: ``def_xy, def_mask, def_age_days, def_features``. Each
        has leading batch dim ``B``. Combine with caller-provided
        ``query_xy, x_n, h_n, opp_idx`` (the last just passed through)
        to call the field's ``forward``.

    Notes
    -----
    * ``def_idx == -1`` (padding) maps through a safe-clamp gather:
      the underlying global slot is the 0th row of ``global_xy``, but
      ``def_mask`` is ``False`` there so downstream consumers don't
      use the value.
    * Anchor dates come from ``cache.config.anchor_dates``; per-shot
      ages are computed in **days** as ``anchor − shot_date``.
    * Feature dimensionality is whatever the artifact carries
      (typically :data:`~shotcloud.features.defense_features.DEFENSE_FEATURE_DIM`).
    """
    if opp_idx.dim() != 1 or snapshot_idx.dim() != 1:
        raise ValueError(
            f"opp_idx and snapshot_idx must be 1-D; got shapes "
            f"{tuple(opp_idx.shape)}, {tuple(snapshot_idx.shape)}"
        )
    if opp_idx.shape[0] != snapshot_idx.shape[0]:
        raise ValueError(
            f"opp_idx and snapshot_idx must have the same length; got "
            f"{opp_idx.shape[0]} and {snapshot_idx.shape[0]}"
        )
    device = opp_idx.device
    cache_device = cache.def_idx.device
    if cache_device != device:
        # Ensure tensors are on the same device as the request. The
        # cache is typically built on CPU; we move the per-row gathers
        # over to whatever device the model lives on.
        opp_idx_dev = opp_idx.to(cache_device)
        snap_idx_dev = snapshot_idx.to(cache_device)
    else:
        opp_idx_dev = opp_idx
        snap_idx_dev = snapshot_idx

    # (B, M_def) gather of indices + mask from the (n_opp, n_snap,
    # M_def) cache tensors.
    def_idx_b = cache.def_idx[opp_idx_dev, snap_idx_dev]  # (B, M_def) int64
    def_mask_b = cache.def_mask[opp_idx_dev, snap_idx_dev]  # (B, M_def) bool

    # Safe-clamp -1 padding to 0 before index_select; mask remains
    # False on those slots so downstream consumers must apply it.
    safe_idx = def_idx_b.clamp_min(0)
    m_def = safe_idx.shape[-1]
    flat = safe_idx.reshape(-1)
    def_xy = cache.global_xy.index_select(0, flat).view(-1, m_def, 2)
    def_dates = cache.global_dates.index_select(0, flat).view(-1, m_def)

    # Per-row anchor dates from the config (kept as a Python tuple
    # in the frozen config). Use snapshot_idx to gather.
    anchor_dates_np = list(cache.config.anchor_dates)
    anchor_tensor = torch.tensor(anchor_dates_np, dtype=torch.int64, device=cache_device)
    anchor_b = anchor_tensor[snap_idx_dev]  # (B,)
    def_age_days = (anchor_b.unsqueeze(-1) - def_dates).to(torch.float32)  # (B, M_def)

    # Defense features: (n_opp, n_snap, D_def) → (B, D_def).
    feat_buf = features.features.to(cache_device)
    def_features = feat_buf[opp_idx_dev, snap_idx_dev]  # (B, D_def)

    # Move everything to the requested device.
    if cache_device != device:
        def_xy = def_xy.to(device)
        def_mask_b = def_mask_b.to(device)
        def_age_days = def_age_days.to(device)
        def_features = def_features.to(device)

    return {
        "def_xy": def_xy,
        "def_mask": def_mask_b,
        "def_age_days": def_age_days,
        "def_features": def_features,
    }


__all__ = [
    "DEFAULT_DEFENSE_BANDWIDTH_FT",
    "DEFAULT_DEFENSE_BETA_INIT",
    "DEFAULT_DEFENSE_QUERY_CHUNK_SIZE",
    "ContinuousAdaptiveDefensiveField",
    "gather_defense_inputs",
]
