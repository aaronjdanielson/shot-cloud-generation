"""Learnable weights for the KDE-product base measure.

Deprecated; retained to reproduce the KDE-product ablation. Superseded by
:class:`~shotcloud.models.continuous_mixture_spatial.ContinuousMixtureSpatial`.

Makes the ``(a_p, a_g, a_0)`` exponents of
:class:`~shotcloud.legacy.product.KDEProduct` **trainable** scalars,
optimized end-to-end alongside the encoder embedding and decoder ``V``
matrix. Each weight is parameterized as
``a_j = softplus(θ_j) = log(1 + exp(θ_j))`` so it stays non-negative
without box constraints, and the spatial decoder is

.. math::

    p_\\theta(c \\mid p) = \\mathrm{softmax}_c\\!\\left[
        a_p \\log \\hat q_p(c)
        + a_g \\log \\hat q_{g(p)}(c)
        + a_0 \\log \\hat q_0(c)
        + u^\\top v_c
    \\right].

Initialization defaults to the hand-picked baseline ``(1.0, 0.3, 0.2)``,
so the first forward pass exactly reproduces the frozen-weights model;
training deviates from that baseline only where doing so reduces the loss.

The weights interact with one another and with the tilt term through the
shared normalization, so the values the optimizer settles on reflect the
factorization as much as the data. They are suitable for prediction, but
**individual weights should not be interpreted as importance scores**.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_INIT_WEIGHTS: dict[str, float] = {"a_p": 1.0, "a_g": 0.3, "a_0": 0.2}


def _invert_softplus(a: float) -> float:
    """Solve ``softplus(θ) = a`` for ``θ``, given ``a > 0``.

    ``softplus(θ) = log(1 + exp(θ)) = a``
    ``=> θ = log(exp(a) - 1) = log(expm1(a))``.

    ``log(expm1(·))`` is numerically stable for the small positive floats
    used as initial weights.
    """
    if a <= 0:
        raise ValueError(f"init weight must be strictly positive (softplus is positive); got {a}")
    return float(np.log(np.expm1(a)))


class LearnableKDEProductWeights(nn.Module):
    """Trainable KDE-product weights via softplus reparameterization.

    Parameters
    ----------
    init_weights : dict[str, float], optional
        Initial values for ``a_p``, ``a_g``, ``a_0``. All must be strictly
        positive (the softplus reparameterization cannot produce 0). Defaults
        to ``{"a_p": 1.0, "a_g": 0.3, "a_0": 0.2}``.

    Notes
    -----
    The module owns three scalar :class:`~torch.nn.Parameter` ``θ`` values.
    The non-negative weights ``a = softplus(θ)`` are exposed via
    :meth:`weights`. Forward takes the per-shot log-densities of the three
    components (player, position, league) and returns a normalized
    ``log q_0`` ready for the decoder.
    """

    def __init__(self, init_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        if init_weights is None:
            init_weights = dict(DEFAULT_INIT_WEIGHTS)
        required = {"a_p", "a_g", "a_0"}
        missing = required - init_weights.keys()
        extra = init_weights.keys() - required
        if missing:
            raise ValueError(f"init_weights missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(
                f"init_weights has unknown keys: {sorted(extra)}. "
                "Only {a_p, a_g, a_0} are supported in v1."
            )

        self.theta_p = nn.Parameter(torch.tensor(_invert_softplus(init_weights["a_p"])))
        self.theta_g = nn.Parameter(torch.tensor(_invert_softplus(init_weights["a_g"])))
        self.theta_0 = nn.Parameter(torch.tensor(_invert_softplus(init_weights["a_0"])))

    def weights(self) -> dict[str, Tensor]:
        """Current non-negative weights ``a_j = softplus(θ_j)``."""
        return {
            "a_p": F.softplus(self.theta_p),
            "a_g": F.softplus(self.theta_g),
            "a_0": F.softplus(self.theta_0),
        }

    def weights_as_floats(self) -> dict[str, float]:
        """Detached float view, useful for logging."""
        with torch.no_grad():
            w = self.weights()
        return {k: float(v) for k, v in w.items()}

    def forward(self, log_qp: Tensor, log_qg: Tensor, log_ql: Tensor) -> Tensor:
        """Compute ``log q_0`` from per-component log-densities.

        Parameters
        ----------
        log_qp : Tensor, shape ``(B, n_cells)``
            Per-shot log raw player density.
        log_qg : Tensor, shape ``(B, n_cells)``
            Per-shot log position-group density.
        log_ql : Tensor, shape ``(n_cells,)`` or ``(B, n_cells)``
            League log density. If 1-D, broadcast over the batch.

        Returns
        -------
        log_q0 : Tensor, shape ``(B, n_cells)``
            Normalized so ``exp(log_q0).sum(dim=-1) == 1``.
        """
        if log_qp.shape != log_qg.shape:
            raise ValueError(
                f"log_qp and log_qg must have the same shape; "
                f"got {tuple(log_qp.shape)} vs {tuple(log_qg.shape)}"
            )
        if log_ql.dim() == 1:
            log_ql = log_ql.unsqueeze(0)
        if log_ql.shape[-1] != log_qp.shape[-1]:
            raise ValueError(
                f"log_ql last-dim must equal n_cells={log_qp.shape[-1]}; got {log_ql.shape[-1]}"
            )

        a_p = F.softplus(self.theta_p)
        a_g = F.softplus(self.theta_g)
        a_0 = F.softplus(self.theta_0)

        log_unnorm = a_p * log_qp + a_g * log_qg + a_0 * log_ql
        log_z = torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
        return log_unnorm - log_z

    def extra_repr(self) -> str:
        with torch.no_grad():
            a = self.weights()
        return f"a_p={float(a['a_p']):.3f}, a_g={float(a['a_g']):.3f}, a_0={float(a['a_0']):.3f}"
