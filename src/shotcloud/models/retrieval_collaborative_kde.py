r"""Retrieval-based support backend for the AC-KDE spatial factor.

:class:`RetrievalCollaborativeKDE` supplies the causal support set of
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`
in the mainline configuration (``--support-backend retrieval``). It
implements the same ``forward_continuous`` contract as
:class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`, but builds
the support from a precomputed
:class:`~shotcloud.models.retrieval_cache.RetrievalCache` instead of a
fixed ``L × R`` analogue grid:

* **Own support**: up to ``own_support_max`` of the target player's
  shots dated before the snapshot anchor, most recent first.
* **Pooled support**: up to ``pooled_support_max`` shots by other players
  within a recency window before the anchor, ranked by trait similarity
  to the target and recency.

Each support slot carries its own shooter identity and stored shot
context :math:`z_j`, and is scored individually,

.. math::

    \ell_j = A_\Theta(p, p_j, x_n) + B_\Theta(x_n, z_j)
             - \lambda_{\mathrm{age}}\,\Delta t_j,

with the same player-level relevance MLP, shot attention, and bandwidth
head as :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`.
Parameters are named identically in both backends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from shotcloud.data import ContextEncoder
from shotcloud.data.context import CONTEXT_DIM
from shotcloud.data.player_traits import TRAIT_DIM, PlayerTraitsTable
from shotcloud.models.collaborative_kde import (
    _H_Z_WARM_INIT_STD,
    _SLOT_LOG1P_M,
    _SLOT_LOG1P_S,
    _SLOT_LOG_DENSITY,
    AlphaPrior,
    CollaborativeContinuousOutputs,
    HzInit,
    ShotAttentionForm,
    _inverse_sigmoid,
)
from shotcloud.models.retrieval_cache import RetrievalCache
from shotcloud.training.dataset import PlayerVocab


class RetrievalCollaborativeKDE(nn.Module):
    r"""Collaborative adaptive KDE support backend with retrieved own/pooled support.

    For a target player :math:`p` at snapshot :math:`t_n`, the support set
    is the concatenation of the cached own block (``own_support_max``
    slots) and pooled block (``pooled_support_max`` slots), so
    ``M = own_support_max + pooled_support_max``. Each valid slot
    :math:`j` with shooter :math:`p_j` receives the logit

    .. math::

        \ell_j = \phi_\theta([u_p, u_{p_j}, u_p - u_{p_j}, u_p \odot u_{p_j}, x_n])
                 + b_{\mathrm{same}}\,[p_j = p]
                 + \gamma \cos(u_p, u_{p_j})
                 + \lambda_M \log(1 + M_{p_j}) + \lambda_S \log(1 + S_{p_j})
                 + g_\theta(x_n, z_j) - \lambda_{\mathrm{age}}\,\Delta t_j,

    where the similarity term is present only with
    ``alpha_prior="similarity"`` and :math:`\Delta t_j` is the shot's age
    in days at the snapshot anchor. The bandwidth :math:`\sigma_p` uses the
    same evidence-dependent formula as
    :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`.

    The cache indices, global shot table, per-shot context, traits, and
    anchor dates are registered as non-persistent buffers, so a saved
    state dict holds only learned parameters; the data buffers are
    rebuilt from ``shots_df`` and ``retrieval_cache`` when the model is
    reconstructed.

    Parameters
    ----------
    retrieval_cache : RetrievalCache
        Precomputed own and pooled support indices per (player, snapshot)
        and the global shot table they index.
    shots_df : pd.DataFrame
        The full shot table the cache was built from (no train/validation
        split; causality is enforced by the cache's anchor-date cut). Used
        to rebuild the per-shot context :math:`z_j`.
    context_encoder : ContextEncoder
        Fitted encoder that maps ``shots_df`` rows to context vectors.
    traits_table : PlayerTraitsTable
        Causal per-(player, snapshot) trait vectors.
    vocab : PlayerVocab
        Player vocabulary shared with ``retrieval_cache`` and
        ``traits_table``.
    phi_hidden_dim, shot_hidden_dim, shot_proj_dim : int
        Hidden and projection widths of the attention MLPs.
    alpha_prior, alpha_prior_scale_init, same_player_bias_init
        Player-level score options.
    shot_attention_form, h_z_init
        Shot-attention form and initialization.
    sigma_min, sigma_max, sigma_init : float
        Bandwidth bounds and initial bandwidth, in feet.

    All hyperparameters above have the same meaning and defaults as in
    :class:`~shotcloud.models.collaborative_kde.CollaborativeKDE`.

    Raises
    ------
    ValueError
        If the trait table does not match the vocabulary or
        :data:`~shotcloud.data.player_traits.TRAIT_DIM`, the bandwidth
        bounds are inconsistent, or ``shots_df`` does not reproduce the
        cache's global shot table.
    """

    sigma_min: Tensor
    sigma_max: Tensor
    sigma_range: Tensor
    global_xy: Tensor
    global_dates: Tensor
    global_shooter_idx: Tensor
    own_idx: Tensor
    own_mask_cache: Tensor
    pooled_idx: Tensor
    pooled_mask_cache: Tensor
    traits: Tensor
    anchor_dates: Tensor
    global_context: Tensor

    def __init__(
        self,
        *,
        retrieval_cache: RetrievalCache,
        shots_df: pd.DataFrame,
        context_encoder: ContextEncoder,
        traits_table: PlayerTraitsTable,
        vocab: PlayerVocab,
        # Scoring hyperparameters — defaults match CollaborativeKDE.
        phi_hidden_dim: int = 64,
        shot_hidden_dim: int = 64,
        shot_proj_dim: int = 32,
        alpha_prior: AlphaPrior = "none",
        alpha_prior_scale_init: float = 2.0,
        same_player_bias_init: float = 0.0,
        shot_attention_form: ShotAttentionForm = "bilinear",
        h_z_init: HzInit = "zero",
        sigma_min: float = 0.75,
        sigma_max: float = 4.0,
        sigma_init: float = 1.5,
    ) -> None:
        super().__init__()

        if traits_table.trait_dim != TRAIT_DIM:
            raise ValueError(
                f"traits_table.trait_dim={traits_table.trait_dim} != TRAIT_DIM={TRAIT_DIM}"
            )
        n_players = len(vocab)
        if traits_table.n_players != n_players:
            raise ValueError(
                f"traits_table.n_players={traits_table.n_players} != len(vocab)={n_players}"
            )

        # ---- σ bounds (needed by the a_0 initialization below) ----
        if not (sigma_min <= sigma_init <= sigma_max):
            raise ValueError(
                f"sigma_min={sigma_min} <= sigma_init={sigma_init} <= "
                f"sigma_max={sigma_max} required"
            )
        self.register_buffer(
            "sigma_min", torch.tensor(float(sigma_min), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_max", torch.tensor(float(sigma_max), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_range",
            torch.tensor(float(sigma_max) - float(sigma_min), dtype=torch.float32),
            persistent=False,
        )
        self.sigma_init = float(sigma_init)

        # ---- Cache buffers (non-persistent; rebuilt from the cache). ----
        self.register_buffer("global_xy", retrieval_cache.global_xy.float(), persistent=False)
        self.register_buffer(
            "global_dates", retrieval_cache.global_dates.to(torch.int64), persistent=False
        )
        self.register_buffer(
            "global_shooter_idx",
            retrieval_cache.global_shooter_idx.to(torch.int64),
            persistent=False,
        )
        self.register_buffer("own_idx", retrieval_cache.own_idx.to(torch.int64), persistent=False)
        self.register_buffer(
            "own_mask_cache", retrieval_cache.own_mask.to(torch.bool), persistent=False
        )
        self.register_buffer(
            "pooled_idx", retrieval_cache.pooled_idx.to(torch.int64), persistent=False
        )
        self.register_buffer(
            "pooled_mask_cache",
            retrieval_cache.pooled_mask.to(torch.bool),
            persistent=False,
        )

        # ---- Traits and anchor dates (non-persistent). ----
        self.register_buffer(
            "traits",
            torch.from_numpy(traits_table.traits.copy()).to(torch.float32),
            persistent=False,
        )
        anchor_dates_np = np.asarray(retrieval_cache.config.anchor_dates, dtype=np.int64)
        self.register_buffer("anchor_dates", torch.from_numpy(anchor_dates_np), persistent=False)

        # ---- Global per-shot context (non-persistent; rebuilt from shots_df). ----
        # The cache is sorted ascending by date and filtered to in-vocab
        # shooters. Apply the same filter+sort to shots_df, transform,
        # and store the resulting (N_global, CONTEXT_DIM) tensor.
        global_context = self._build_global_context(shots_df, context_encoder, vocab)
        if global_context.shape[0] != retrieval_cache.global_xy.shape[0]:
            raise ValueError(
                f"global_context length {global_context.shape[0]} does not match "
                f"cache global length {retrieval_cache.global_xy.shape[0]}; the "
                f"shots_df has changed since the cache was built. Pass the "
                f"**full** shots_df (no train/val split) — the cache stores all "
                f"in-vocab shots sorted by date, and the snapshot-anchor causal "
                f"slice handles val-leak prevention. (If the shots file itself "
                f"changed, the shots_fingerprint mismatch will rebuild the cache.)"
            )
        self.register_buffer("global_context", global_context, persistent=False)

        self._own_max = int(retrieval_cache.own_idx.shape[-1])
        self._pool_max = int(retrieval_cache.pooled_idx.shape[-1])
        self._M = self._own_max + self._pool_max

        # ===========================================================
        # Trained submodules — named as in CollaborativeKDE so parameter
        # names are shared across backends.
        # ===========================================================

        # A_Θ: relevance MLP on [u_self, u_other, u_self - u_other,
        #                        u_self ⊙ u_other, x_n].
        phi_in = 4 * TRAIT_DIM + CONTEXT_DIM
        phi_out_layer = nn.Linear(phi_hidden_dim, 1)
        nn.init.zeros_(phi_out_layer.weight)
        nn.init.zeros_(phi_out_layer.bias)
        self.phi = nn.Sequential(
            nn.Linear(phi_in, phi_hidden_dim),
            nn.GELU(),
            phi_out_layer,
        )

        # Player-level scalars.
        self.b_same = nn.Parameter(torch.tensor(float(same_player_bias_init), dtype=torch.float32))
        self.lambda_M = nn.Parameter(torch.zeros(()))
        self.lambda_S = nn.Parameter(torch.zeros(()))
        self.alpha_prior: AlphaPrior = alpha_prior
        if alpha_prior == "similarity":
            self.gamma_sim = nn.Parameter(
                torch.tensor(float(alpha_prior_scale_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma_sim", torch.zeros((), dtype=torch.float32), persistent=False
            )

        # B_Θ: bilinear (default) or concat shot attention.
        self.shot_attention_form: ShotAttentionForm = shot_attention_form
        if shot_attention_form == "bilinear":
            f_out = nn.Linear(shot_hidden_dim, shot_proj_dim)
            h_out = nn.Linear(shot_hidden_dim, shot_proj_dim)
            if h_z_init == "zero":
                nn.init.zeros_(h_out.weight)
            else:  # warm
                nn.init.normal_(h_out.weight, mean=0.0, std=_H_Z_WARM_INIT_STD)
            nn.init.zeros_(h_out.bias)
            self.f_x = nn.Sequential(
                nn.Linear(CONTEXT_DIM, shot_hidden_dim),
                nn.GELU(),
                f_out,
            )
            self.h_z = nn.Sequential(
                nn.Linear(CONTEXT_DIM, shot_hidden_dim),
                nn.GELU(),
                h_out,
            )
            self._shot_proj_dim = int(shot_proj_dim)
            self.h_z_init: HzInit = h_z_init
        else:
            g_in = 2 * CONTEXT_DIM
            g_out_layer = nn.Linear(shot_hidden_dim, 1)
            nn.init.zeros_(g_out_layer.weight)
            nn.init.zeros_(g_out_layer.bias)
            self.g = nn.Sequential(
                nn.Linear(g_in, shot_hidden_dim),
                nn.GELU(),
                g_out_layer,
            )
            self._shot_proj_dim = 0
            self.h_z_init = h_z_init

        # Shot-level learned recency decay rate.
        self.lambda_age = nn.Parameter(torch.zeros(()))

        # σ-head scalars (same formula as CollaborativeKDE).
        if self.sigma_range.item() > 0:
            a0_init = _inverse_sigmoid(
                (self.sigma_init - float(self.sigma_min)) / float(self.sigma_range)
            )
        else:
            a0_init = 0.0
        self.a_0 = nn.Parameter(torch.tensor(a0_init, dtype=torch.float32))
        self.a_M = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.a_S = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.a_R = nn.Parameter(torch.zeros(()))

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _build_global_context(
        shots_df: pd.DataFrame, context_encoder: ContextEncoder, vocab: PlayerVocab
    ) -> Tensor:
        """Return the ``(N_global, CONTEXT_DIM)`` float32 per-shot context table.

        Filters ``shots_df`` to shooters in the vocabulary and sorts it
        stably by date, matching the cache's global ordering, before
        applying the context encoder.
        """
        id_to_idx = {pid: i for i, pid in enumerate(vocab.ids)}
        pid_str = shots_df["player_id"].astype(str).to_numpy()
        keep = np.array([p in id_to_idx for p in pid_str], dtype=bool)
        sub = shots_df.iloc[keep].reset_index(drop=True)
        date_ord = pd.to_datetime(sub["date"]).to_numpy(dtype="datetime64[D]").astype(np.int64)
        order = np.argsort(date_ord, kind="stable")
        sub = sub.iloc[order].reset_index(drop=True)
        ctx = context_encoder.transform(sub)  # (N, CONTEXT_DIM) ndarray
        return torch.from_numpy(np.asarray(ctx, dtype=np.float32))

    @property
    def own_support_max(self) -> int:
        """Number of own-support slots per row."""
        return self._own_max

    @property
    def pooled_support_max(self) -> int:
        """Number of pooled-support slots per row."""
        return self._pool_max

    @property
    def M(self) -> int:  # noqa: N802 — matches the model notation
        """Total support slots per row, ``own_support_max + pooled_support_max``."""
        return self._M

    def _compute_sigma(self, player_idx: Tensor, snapshot_idx: Tensor) -> Tensor:
        """Per-target bandwidth σ_p(t_n), as in ``CollaborativeKDE._compute_sigma``."""
        u_self = self.traits[player_idx, snapshot_idx]
        log1p_M = u_self[..., _SLOT_LOG1P_M]
        log1p_S = u_self[..., _SLOT_LOG1P_S]
        log_density = u_self[..., _SLOT_LOG_DENSITY]
        z = (
            self.a_0
            - torch.nn.functional.softplus(self.a_M) * log1p_M
            - torch.nn.functional.softplus(self.a_S) * log1p_S
            + self.a_R * log_density
        )
        sigma: Tensor = self.sigma_min + self.sigma_range * torch.sigmoid(z)
        return sigma

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward_continuous(
        self,
        player_idx: Tensor,
        snapshot_idx: Tensor,
        x_n_raw: Tensor,
        x_n: Tensor,
    ) -> CollaborativeContinuousOutputs:
        """Gather the retrieved support set and score each support shot.

        Parameters
        ----------
        player_idx : Tensor of shape ``(B,)`` int64
            Vocabulary index of the target player.
        snapshot_idx : Tensor of shape ``(B,)`` int64
            Causal snapshot index of the shot's game.
        x_n_raw : Tensor of shape ``(B, CONTEXT_DIM)``
            Raw context; accepted for interface compatibility and unused.
        x_n : Tensor of shape ``(B, CONTEXT_DIM)``
            Learned context vector.

        Returns
        -------
        CollaborativeContinuousOutputs
            Support set of width ``M`` with the own block first. The
            ``analogue_idx``, ``alpha_scores``, and ``beta_scores``
            diagnostics are empty tensors for this backend.
        """
        del x_n_raw  # accepted for interface compatibility; unused
        b = player_idx.shape[0]

        # ---- 1. Gather per-(player, snapshot) retrieved indices + masks. ----
        own_idx_b = self.own_idx[player_idx, snapshot_idx]  # (B, own_max)
        own_mask_b = self.own_mask_cache[player_idx, snapshot_idx]  # (B, own_max)
        pool_idx_b = self.pooled_idx[player_idx, snapshot_idx]  # (B, pool_max)
        pool_mask_b = self.pooled_mask_cache[player_idx, snapshot_idx]  # (B, pool_max)

        support_idx = torch.cat([own_idx_b, pool_idx_b], dim=-1)  # (B, M)
        # own_mask in the concatenated layout has True only on the own block.
        own_mask = torch.cat([own_mask_b, torch.zeros_like(pool_mask_b)], dim=-1)  # (B, M)
        pooled_mask = torch.cat([torch.zeros_like(own_mask_b), pool_mask_b], dim=-1)  # (B, M)
        support_mask = own_mask | pooled_mask  # (B, M)

        # ---- 2. Safe gather (-1 → 0; mask carries validity). ----
        safe_idx = support_idx.clamp_min(0)
        support_xy = self.global_xy.index_select(0, safe_idx.reshape(-1)).view(b, self._M, 2)
        support_dates = self.global_dates.index_select(0, safe_idx.reshape(-1)).view(b, self._M)
        support_shooter = self.global_shooter_idx.index_select(0, safe_idx.reshape(-1)).view(
            b, self._M
        )
        support_context = self.global_context.index_select(0, safe_idx.reshape(-1)).view(
            b, self._M, CONTEXT_DIM
        )

        # ---- 3. A_Θ per shot ----
        target_trait = self.traits[player_idx, snapshot_idx]  # (B, TRAIT_DIM)
        snap_b_m = snapshot_idx.view(b, 1).expand(-1, self._M)
        shooter_trait = self.traits[support_shooter, snap_b_m]  # (B, M, TRAIT_DIM)
        target_bcast = target_trait.unsqueeze(1).expand(-1, self._M, -1)
        phi_in = torch.cat(
            [
                target_bcast,
                shooter_trait,
                target_bcast - shooter_trait,
                target_bcast * shooter_trait,
                x_n.unsqueeze(1).expand(-1, self._M, -1),
            ],
            dim=-1,
        )  # (B, M, 4*T + C)
        phi_scores = self.phi(phi_in).squeeze(-1)  # (B, M)
        is_self = support_shooter == player_idx.view(b, 1)
        phi_scores = phi_scores + self.b_same * is_self.to(phi_scores.dtype)
        if self.alpha_prior == "similarity":
            t_norm = target_trait.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (B, 1)
            s_norm = shooter_trait.norm(dim=-1).clamp_min(1e-12)  # (B, M)
            cos_sim = (target_trait.unsqueeze(1) * shooter_trait).sum(dim=-1) / (t_norm * s_norm)
            alpha_scores_per_shot = phi_scores + self.gamma_sim * cos_sim
        else:
            alpha_scores_per_shot = phi_scores

        # Evidence-volume prior on the shooter: the L×R term
        # ``λ_M log1p_M + λ_S log1p_S``, applied per support slot since
        # the retrieval layout has no per-analogue score A_l.
        log1p_M_other = shooter_trait[..., _SLOT_LOG1P_M]  # (B, M)
        log1p_S_other = shooter_trait[..., _SLOT_LOG1P_S]
        alpha_scores_per_shot = (
            alpha_scores_per_shot + self.lambda_M * log1p_M_other + self.lambda_S * log1p_S_other
        )

        # ---- 4. B_Θ per shot ----
        if self.shot_attention_form == "bilinear":
            f = self.f_x(x_n)  # (B, P)
            h = self.h_z(support_context)  # (B, M, P)
            beta_scores_per_shot = (f.unsqueeze(1) * h).sum(dim=-1)  # (B, M)
        else:
            x_bcast = x_n.unsqueeze(1).expand(-1, self._M, -1)
            g_in = torch.cat([x_bcast, support_context], dim=-1)
            beta_scores_per_shot = self.g(g_in).squeeze(-1)  # (B, M)

        # ---- 5. Recency + assemble logits ----
        anchor_b = self.anchor_dates[snapshot_idx]  # (B,)
        delta_days = (anchor_b.view(b, 1) - support_dates).to(torch.float32)
        support_logits = alpha_scores_per_shot + beta_scores_per_shot - self.lambda_age * delta_days

        # ---- 6. σ (per-target) ----
        sigma = self._compute_sigma(player_idx, snapshot_idx)

        # ---- 7. Return ----
        # analogue_idx / alpha_scores / beta_scores are L×R-specific
        # diagnostics; they are empty (L = 0) tensors here so consumers
        # need no None checks.
        empty_l = torch.empty(b, 0, dtype=torch.int64, device=support_xy.device)
        empty_l_float = torch.empty(b, 0, dtype=torch.float32, device=support_xy.device)
        empty_l_r = torch.empty(b, 0, 0, dtype=torch.float32, device=support_xy.device)
        return CollaborativeContinuousOutputs(
            support_xy=support_xy,
            support_logits=support_logits,
            sigma=sigma,
            support_mask=support_mask,
            own_mask=own_mask,
            support_shooter=support_shooter,
            analogue_idx=empty_l,
            alpha_scores=empty_l_float,
            beta_scores=empty_l_r,
        )


__all__ = ["RetrievalCollaborativeKDE"]
