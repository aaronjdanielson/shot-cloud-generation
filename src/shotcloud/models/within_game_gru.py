"""Within-game shot-state GRU for the G1 residual extension (paper §10).

G1 augments the B2 residual tilt with a learned causal state over the
target player's prior shots in the same game:

.. math::

    g_r = \\operatorname{GRU}(g_{r-1}, a_{r-1}),

where :math:`a_{r-1}` is the previous shot's normalized feature vector
(coordinate, zone, distance, time gap, period) and :math:`g_r` is fed
into the residual encoder alongside :math:`x_n`, :math:`h_{n,r}`,
:math:`u_{p,t}`, and :math:`c_{p,t}`.

Architecture (locked first-cut):

* One-layer :class:`torch.nn.GRU`, hidden_size ``H`` (default 16).
* Input is the padded per-prior-shot feature tensor of shape
  ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)`` together with a
  per-row length tensor of shape ``(B,)``.
* Hidden state is initialized to zero per row.
* Final layer is a linear projection ``nn.Linear(H, gru_out_dim,
  bias=False)`` whose weight is zero-initialized so the GRU output
  contributes exactly zero at step 0. This is the load-bearing
  invariant ``G1 ≡ B2`` at initialization.

Notes:

* Rows with ``length == 0`` (first shot of a player-game) bypass the
  GRU entirely and return the zero output directly. ``rnn.pack_padded_sequence``
  rejects zero-length items, so we handle them outside the GRU call.
* Output shape: ``(B, gru_out_dim)``.
"""

from __future__ import annotations

from torch import Tensor, nn

from shotcloud.data.within_game_history import MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM


class WithinGameGRU(nn.Module):
    """Causal within-game shot-state encoder for the G1 residual.

    Parameters
    ----------
    hidden_dim : int, default 16
        GRU hidden width. The first-cut spec uses 16; 32 is reserved as
        a follow-up only if 16 helps.
    out_dim : int, default 8
        Output dimension of the linear projection fed into the residual
        encoder. Match this to the residual encoder's residual rank to
        keep the existing zero-init pipeline intact.

    Forward
    -------
    ``forward(prior_seq, prior_lengths) -> Tensor``
        * ``prior_seq`` shape ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)``.
        * ``prior_lengths`` shape ``(B,)`` int64, valid prior count per row.
        * Returns ``(B, out_dim)`` — zero at initialization for every row.
    """

    def __init__(self, hidden_dim: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_dim = WITHIN_GAME_SEQ_DIM
        self.max_prior = MAX_PRIOR_SHOTS

        self.gru = nn.GRU(
            input_size=WITHIN_GAME_SEQ_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=True,
        )
        # Load-bearing zero-init: the projection from the GRU state to
        # the residual feature space is zero, so the G1 augmentation
        # contributes 0 at step 0 for every row regardless of inputs.
        # Likelihood gradients then activate it only if causal in-game
        # history adds information beyond the existing residual inputs.
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, prior_seq: Tensor, prior_lengths: Tensor) -> Tensor:
        if prior_seq.dim() != 3 or prior_seq.shape[2] != self.input_dim:
            raise ValueError(
                f"prior_seq must have shape (B, MAX_PRIOR_SHOTS, {self.input_dim}); "
                f"got {tuple(prior_seq.shape)}"
            )
        if prior_lengths.dim() != 1 or prior_lengths.shape[0] != prior_seq.shape[0]:
            raise ValueError(
                f"prior_lengths must have shape (B,) matching prior_seq's batch dim; "
                f"got {tuple(prior_lengths.shape)}, expected ({prior_seq.shape[0]},)"
            )

        device = prior_seq.device
        b = prior_seq.shape[0]
        out = prior_seq.new_zeros((b, self.out_dim))
        # First-shot rows (length 0) bypass the GRU entirely; their
        # output is the zero vector that projection would produce.
        nonzero = (prior_lengths > 0).cpu()
        if not nonzero.any():
            return out

        seq_nz = prior_seq[nonzero]
        len_nz = prior_lengths[nonzero].cpu()
        # ``pack_padded_sequence`` requires int64 lengths on CPU.
        packed = nn.utils.rnn.pack_padded_sequence(
            seq_nz, len_nz, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n shape: (num_layers=1, n_nonzero, hidden_dim) → (n_nonzero, hidden_dim).
        state = h_n.squeeze(0)
        proj_nz = self.proj(state)
        # Scatter the projected outputs back into the full batch order.
        nonzero_idx = nonzero.nonzero(as_tuple=False).squeeze(-1).to(device)
        out[nonzero_idx] = proj_nz
        return out


__all__ = ["WithinGameGRU"]
