"""Conditional Gibbs spatial decoder over court cells.

Deprecated; retained to reproduce the grid-cell decoder ablations. Superseded
by :class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`,
which evaluates a continuous kernel mixture instead of a softmax over cells.

Composes the three log-energy terms produced by the upstream
modules into the spatial distribution

.. math::

    p_\\Theta(c \\mid x_n, \\mathrm{opp}_n, h_{n,i}, \\tau_{n,i})
    = \\frac{\\exp\\{-E_\\Theta(c)\\}}
           {\\sum_{c'} \\exp\\{-E_\\Theta(c')\\}}

with energy

.. math::

    E_\\Theta(c) = -\\log q^{\\mathrm{off}}(c)
                  - \\log a_\\delta(c)
                  - r_\\theta(c).

The decoder itself is **a composition, not a learnable layer**: it
holds references to the four upstream modules (offensive prior,
defensive field, residual encoder, tilt decoder) and runs their
forwards in sequence. All learnable parameters live on the
upstream modules; this class adds none of its own.

Construction contract
---------------------

The four submodules are constructed independently with their own
data sources (causal pools, snapshot store, vocabs) and threaded
through the constructor. The decoder validates only that they
agree on grid size and on the residual rank.

Forward contract
----------------

``forward(player_idx, opp_idx, snapshot_idx, x_n, ...)`` returns
``(B, n_cells)`` log-probabilities. ``return_components=True``
additionally returns a :class:`GibbsDecoderOutputs` dataclass with
the per-component intermediates that are useful for ablations,
diagnostics, and structural-regularization losses.

Zero-init invariant
-------------------

When :attr:`tilt_decoder.V` is initialized to zero (the default),
the residual term :math:`r_\\theta(c) = u_\\theta(x_n)^\\top v_c` is
zero for every cell. The Gibbs decoder then collapses to

.. math::

    p(c) = \\mathrm{softmax}_c[\\log q^{\\mathrm{off}}(c) + \\log a_\\delta(c)],

which is itself a normalized geometric mean of the offensive prior
and the defensive feasibility field. No information from the
neural residual contaminates the geometry at initialization; training
gradually opens it up via :math:`V`.

Defensive uniform fallback
--------------------------

When an opponent has no causal history at the current anchor,
:meth:`AdaptiveDefensiveField.forward` returns
:math:`\\log a_\\delta(c) = -\\log C` (uniform). That's a constant
across cells, so it slides through the softmax untouched: rows
without defensive history reduce cleanly to

.. math::

    p(c) = \\mathrm{softmax}_c[\\log q^{\\mathrm{off}}(c) + r_\\theta(c)].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from shotcloud.legacy_pivot.adaptive_defensive import AdaptiveDefensiveField
from shotcloud.legacy_pivot.adaptive_prior import AdaptiveOffensivePrior
from shotcloud.legacy_pivot.tilt_decoder import LowRankTiltDecoder
from shotcloud.models.collaborative_kde import CollaborativeKDE
from shotcloud.models.context_residual import ContextResidualEncoder

# Either offensive-prior backbone: the ω-gated self/archetype mixture or
# the collaborative KDE. They have different forward signatures; the
# decoder dispatches on the type at runtime.
OffensivePriorBackbone = AdaptiveOffensivePrior | CollaborativeKDE


@dataclass(frozen=True)
class GibbsDecoderOutputs:
    """Per-component intermediates from one Gibbs decoder forward pass.

    All tensors are batched along axis 0 and live on the same device
    as the inputs. Useful for downstream losses (entropy/ESS
    regularizers on the relevance softmaxes), component ablations
    (zero out one component, recompose), and visualization.

    Attributes
    ----------
    log_q_off : Tensor of shape ``(B, n_cells)``
        Log offensive density :math:`\\log q^{\\mathrm{off}}(c)`.
    log_a_delta : Tensor of shape ``(B, n_cells)``
        Log defensive feasibility field
        :math:`\\log a_\\delta(c)`.
    r_theta : Tensor of shape ``(B, n_cells)``
        Residual energy correction
        :math:`u_\\theta(x_n)^\\top v_c`.
    energy_neg : Tensor of shape ``(B, n_cells)``
        Sum ``log_q_off + log_a_delta + r_theta`` (the negative
        Gibbs energy; equivalent up to sign).
    omega : Tensor of shape ``(B,)``
        Offensive prior's ESS-shrinkage gate :math:`\\omega_n`.
    pi_off : Tensor of shape ``(B, max_N_off)``
        Offensive relevance softmax (zero on rows without causal
        self-history).
    pi_def : Tensor of shape ``(B, max_N_def)``
        Defensive relevance softmax (zero on rows without causal
        opponent history).
    has_def_history : Tensor of shape ``(B,)``
        Whether the row had any causal defensive history; rows
        without history get the uniform fallback inside
        :class:`AdaptiveDefensiveField`.
    """

    log_q_off: Tensor
    log_a_delta: Tensor
    r_theta: Tensor
    energy_neg: Tensor
    omega: Tensor
    pi_off: Tensor
    pi_def: Tensor
    has_def_history: Tensor
    #: Rich per-prior intermediates. ``CollaborativeOutputs`` for the
    #: collaborative-KDE path (carries ``alpha``, ``beta``, ``sigma``,
    #: etc. — used by trainer-side entropy diagnostics); ``None`` for
    #: the ``AdaptiveOffensivePrior`` path. Typed as ``object``
    #: so this module doesn't acquire a downward import; downstream
    #: consumers use ``isinstance(prior_components, CollaborativeOutputs)``.
    prior_components: object | None = None


class ConditionalGibbsDecoder(nn.Module):
    """Composes ``q_off``, ``a_δ``, ``r_θ`` into the Gibbs spatial distribution.

    Combines a required offensive prior with an *optional* defensive field and
    an *optional* low-rank residual tilt, producing the conditional
    log-probability ``log_softmax(log q_off + [log a_δ] + [r_θ])``
    where each bracketed term is included only when its module is
    supplied. A single constructor covers four ablation regimes:

    * offense only: ``offensive_prior`` alone;
    * offense + defense: add ``defensive_field``;
    * offense + residual: add ``residual_encoder`` and
      ``tilt_decoder`` (both required when either is present);
    * full Gibbs decoder: all four.

    Parameters
    ----------
    offensive_prior : AdaptiveOffensivePrior or CollaborativeKDE
        Produces ``log q_off`` for each row. :class:`AdaptiveOffensivePrior`
        also produces ``π_off`` and ``ω``; for :class:`CollaborativeKDE`
        those outputs are zero-filled placeholders.
    defensive_field : AdaptiveDefensiveField, optional
        Produces ``log a_δ``, ``π_def``, ``has_history`` for each row.
        Omitted when training an offense-only configuration; the
        spatial logits then become
        ``log_softmax(log q_off + [r_θ])``.
    residual_encoder : ContextResidualEncoder, optional
        Produces ``u_θ(x_n)`` of shape ``(B, rank)``. Must be
        provided iff ``tilt_decoder`` is provided.
    tilt_decoder : LowRankTiltDecoder, optional
        Produces ``r_θ(c) = u_θ(x_n)^T v_c`` from ``u`` via
        :meth:`LowRankTiltDecoder.tilt`. Initialize with
        ``zero_init=True`` (default) for the zero-init invariant.
        Must be provided iff ``residual_encoder`` is provided.

    Notes
    -----
    The decoder owns no parameters of its own — all learnable state
    lives on the submodules. ``self.parameters()`` returns exactly
    the union of the supplied submodules' parameters.
    """

    def __init__(
        self,
        offensive_prior: OffensivePriorBackbone,
        defensive_field: AdaptiveDefensiveField | None = None,
        residual_encoder: ContextResidualEncoder | None = None,
        tilt_decoder: LowRankTiltDecoder | None = None,
    ) -> None:
        super().__init__()

        # Validate grid agreement on every supplied spatial-output module.
        if defensive_field is not None and offensive_prior.n_cells != defensive_field.n_cells:
            raise ValueError(
                f"offensive_prior.n_cells={offensive_prior.n_cells} != "
                f"defensive_field.n_cells={defensive_field.n_cells}; "
                f"both must operate on the same court grid"
            )

        # Residual encoder and tilt decoder are an inseparable pair.
        if (residual_encoder is None) != (tilt_decoder is None):
            raise ValueError("residual_encoder and tilt_decoder must be both provided or both None")
        if residual_encoder is not None and tilt_decoder is not None:
            if tilt_decoder.n_cells != offensive_prior.n_cells:
                raise ValueError(
                    f"tilt_decoder.n_cells={tilt_decoder.n_cells} != "
                    f"offensive_prior.n_cells={offensive_prior.n_cells}"
                )
            if residual_encoder.rank != tilt_decoder.rank:
                raise ValueError(
                    f"residual_encoder.rank={residual_encoder.rank} != "
                    f"tilt_decoder.rank={tilt_decoder.rank}; the two factors "
                    f"of the low-rank residual must agree"
                )

        self.offensive_prior = offensive_prior
        self.defensive_field = defensive_field
        self.residual_encoder = residual_encoder
        self.tilt_decoder = tilt_decoder
        self._n_cells = int(offensive_prior.n_cells)

    @property
    def has_defense(self) -> bool:
        return self.defensive_field is not None

    @property
    def has_residual(self) -> bool:
        return self.residual_encoder is not None

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def forward(
        self,
        player_idx: Tensor,
        opp_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
        h_n: Tensor | None = None,
        games_ago_off: Tensor | None = None,
        games_ago_def: Tensor | None = None,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, GibbsDecoderOutputs]:
        """Compute log-probabilities over court cells.

        Parameters
        ----------
        player_idx : LongTensor of shape ``(B,)``
            Per-row player index into the offensive prior's vocab.
        opp_idx : LongTensor of shape ``(B,)``
            Per-row opponent index into the defensive field's vocab.
            Ignored when no defensive field is configured (the trainer
            may pass sentinel zeros in that case).
        snapshot_idx : LongTensor of shape ``(B,)``
            Per-row snapshot index for both upstream modules' causal
            date masks.
        x_n_raw : Tensor of shape ``(B, CONTEXT_DIM)``
            Raw context vector :math:`\\tilde x_n` from
            :class:`ContextEncoder`. Threaded to the structured
            :class:`RelevanceScore` in both the offensive prior and the
            defensive field so the named slices keep their documented
            meaning.
        x_n : Tensor of shape ``(B, CONTEXT_DIM)``
            Learned context :math:`x_n = f_{\\mathrm{ctx}}(\\tilde
            x_n)`. Consumed by the linear-in-context heads:
            :class:`ArchetypeMixture` (inside the offensive prior) and
            :class:`ContextResidualEncoder`.
        games_ago_off, games_ago_def : Tensor, optional
            Per-row recency features for the offensive and defensive
            relevance scores respectively. Threaded through if given;
            both upstream modules accept ``None``.
        return_components : bool, default False
            If True, return ``(log_probs, GibbsDecoderOutputs)``;
            otherwise return ``log_probs`` only. The components
            dataclass uses zero tensors for any omitted submodule.

        Returns
        -------
        Tensor of shape ``(B, n_cells)``
            Log-probabilities :math:`\\log p(c)` for each row.
        """
        # Offensive prior dispatch. AdaptiveOffensivePrior returns
        # (log_q, π, ω); CollaborativeKDE returns log_q alone (with
        # optional rich components). Both produce the same (B, n_cells)
        # log_q for the energy sum; placeholder π / ω fill the
        # GibbsDecoderOutputs fields for the collaborative path.
        prior_components: object | None = None
        if isinstance(self.offensive_prior, CollaborativeKDE):
            # When components are requested downstream (e.g. for
            # H(α) / H(β) diagnostics) we ask the collaborative path
            # for them too. Otherwise we take the cheaper log-only
            # forward — gradient path is identical either way.
            if return_components:
                log_q_off_or_pair = self.offensive_prior(
                    player_idx,
                    snapshot_idx,
                    x_n_raw,
                    x_n,
                    return_components=True,
                )
                assert isinstance(log_q_off_or_pair, tuple)
                log_q_off, prior_components = log_q_off_or_pair
            else:
                log_q_off = self.offensive_prior(player_idx, snapshot_idx, x_n_raw, x_n)
            assert isinstance(log_q_off, Tensor)
            # Collaborative architecture has no π, ω — use zero-filled
            # placeholders so GibbsDecoderOutputs keeps the same fields.
            pi_off = torch.zeros(
                log_q_off.shape[0], 1, dtype=log_q_off.dtype, device=log_q_off.device
            )
            omega = torch.zeros(log_q_off.shape[0], dtype=log_q_off.dtype, device=log_q_off.device)
        else:
            log_q_off, pi_off, omega = self.offensive_prior(
                player_idx, snapshot_idx, x_n_raw, x_n, games_ago=games_ago_off
            )
        energy_neg = log_q_off

        # Defensive feasibility (optional). When absent, behaves as
        # log a_δ ≡ 0 — i.e., uniform feasibility, which slides
        # through the softmax untouched.
        if self.defensive_field is not None:
            log_a_delta, pi_def, has_def_history = self.defensive_field(
                opp_idx, snapshot_idx, x_n_raw, games_ago=games_ago_def
            )
            energy_neg = energy_neg + log_a_delta
        else:
            log_a_delta = torch.zeros_like(log_q_off)
            pi_def = torch.zeros(
                log_q_off.shape[0], 1, dtype=log_q_off.dtype, device=log_q_off.device
            )
            has_def_history = torch.zeros(
                log_q_off.shape[0], dtype=torch.bool, device=log_q_off.device
            )

        # Residual tilt (optional). When V=0 (tilt_decoder zero-init)
        # this is zero regardless of u; the trainer activates it
        # gradually via the V gradient.
        if self.residual_encoder is not None and self.tilt_decoder is not None:
            # When the encoder consumes within-game history, h_n must
            # be provided. Otherwise (within_game_dim=0) h_n is ignored.
            if self.residual_encoder.within_game_dim > 0:
                if h_n is None:
                    raise ValueError(
                        "residual_encoder.within_game_dim > 0 → h_n is required; "
                        "pass the per-shot within-game-history tensor from the dataset"
                    )
                u = self.residual_encoder(x_n, h_n)
            else:
                u = self.residual_encoder(x_n)
            r_theta = self.tilt_decoder.tilt(u)
            energy_neg = energy_neg + r_theta
        else:
            r_theta = torch.zeros_like(log_q_off)

        log_probs = torch.log_softmax(energy_neg, dim=-1)

        if not return_components:
            return log_probs

        components = GibbsDecoderOutputs(
            log_q_off=log_q_off,
            log_a_delta=log_a_delta,
            r_theta=r_theta,
            energy_neg=energy_neg,
            omega=omega,
            pi_off=pi_off,
            pi_def=pi_def,
            has_def_history=has_def_history,
            prior_components=prior_components,
        )
        return log_probs, components

    def extra_repr(self) -> str:
        return f"n_cells={self._n_cells}"
