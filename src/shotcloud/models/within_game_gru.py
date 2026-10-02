"""Within-game shot-state GRU for the residual tilt.

:class:`WithinGameGRU` encodes the target player's earlier shots in the
same game into a learned causal state,

.. math::

    g_r = \\operatorname{GRU}(g_{r-1}, a_{r-1}),

where :math:`a_{r-1}` is the previous shot's normalized feature vector
(coordinate, zone, distance, time gap, period). The projected final
state is added to the residual-encoder output :math:`u_\\theta`, so it
tilts the support logits through the same low-rank residual. It is an
extension of the residual evaluated as an ablation; only shots before
the current one enter the sequence.

Architecture:

* One-layer :class:`torch.nn.GRU` with hidden size ``H`` (default 16)
  and a zero initial state.
* Input is the padded per-prior-shot feature tensor of shape
  ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)`` together with a per-row
  length tensor of shape ``(B,)``.
* The output projection ``nn.Linear(H, out_dim, bias=False)`` is
  zero-initialized, so the GRU contributes exactly zero at
  initialization and the model starts from the residual without the
  GRU.

Rows with ``length == 0`` (the first shot of a player-game) bypass the
GRU and return zeros, because ``pack_padded_sequence`` rejects
zero-length sequences.
"""

from __future__ import annotations

from torch import Tensor, nn

from shotcloud.data.within_game_history import MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM


class WithinGameGRU(nn.Module):
    """Causal within-game shot-state encoder for the residual tilt.

    Parameters
    ----------
    hidden_dim : int, default 16
        GRU hidden width.
    out_dim : int, default 8
        Output dimension of the linear projection. Must equal the rank of
        the residual encoder, whose output the projection is added to.
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
        # Zero-init the projection so the GRU contributes 0 at
        # initialization for every row regardless of inputs; the
        # likelihood gradient moves it away from zero only if in-game
        # history adds information beyond the other residual inputs.
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, prior_seq: Tensor, prior_lengths: Tensor) -> Tensor:
        """Encode each row's earlier in-game shots.

        Parameters
        ----------
        prior_seq : Tensor of shape ``(B, MAX_PRIOR_SHOTS, WITHIN_GAME_SEQ_DIM)``
            Padded feature sequence of the player's earlier shots in the
            game.
        prior_lengths : Tensor of shape ``(B,)``, int64
            Number of valid prior shots per row.

        Returns
        -------
        Tensor of shape ``(B, out_dim)``
            Projected final GRU state; zero for rows with no prior shots
            and, at initialization, for every row.
        """
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
