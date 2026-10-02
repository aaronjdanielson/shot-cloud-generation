r"""Learned context embedding :math:`x_n = f_{\text{ctx}}(\tilde x_n)`.

:class:`ContextMLP` maps the raw per-shot feature vector
:math:`\tilde x_n \in \mathbb R^{D_{\text{raw}}}` produced by
:class:`shotcloud.data.ContextEncoder` to a learned representation
:math:`x_n \in \mathbb R^{D_{\text{ctx}}}`. The learned :math:`x_n` feeds
the linear-in-context heads (count head, timing head, residual-tilt
encoder, pooling gate). Structured similarity scores such as
:class:`~shotcloud.models.RelevanceScore` consume the raw
:math:`\tilde x_n` instead, so that their named feature slices stay
interpretable.

By default the MLP runs in residual mode with a zero-initialized output
projection, so :math:`f_{\text{ctx}}(\tilde x_n) = \tilde x_n` exactly at
initialization: every downstream model starts from its raw-feature
behavior, and training learns a small additive deformation.

Capacity is intentionally small. With the defaults
(``input_dim = output_dim = CONTEXT_DIM = 27``, ``hidden_dim = 64``, one
GELU) the module has :math:`2 \cdot 27 \cdot 64 + 64 + 27 = 3{,}547`
parameters.
"""

from __future__ import annotations

from torch import Tensor, nn


class ContextMLP(nn.Module):
    r"""One-hidden-layer MLP mapping :math:`\tilde x_n \to x_n`.

    Parameters
    ----------
    input_dim : int
        Raw context dimension :math:`D_{\text{raw}}`. Downstream modules
        expect :data:`shotcloud.data.context.CONTEXT_DIM`.
    hidden_dim : int, default 64
        Width of the hidden layer. Small by design: the MLP is a mild
        deformation of the raw features, not a full feature encoder.
    output_dim : int, optional
        Output dimension :math:`D_{\text{ctx}}`. Defaults to
        ``input_dim``; must equal ``input_dim`` when ``residual=True``.
    residual : bool, default True
        If True, return :math:`\tilde x_n + h(\tilde x_n)` with the final
        projection zero-initialized, so the module is the identity at
        initialization. If False, return :math:`h(\tilde x_n)` with
        PyTorch's default initialization.
    dropout : float, default 0.0
        Dropout rate applied after the hidden activation.

    Raises
    ------
    ValueError
        If a dimension is non-positive, ``dropout`` is outside
        ``[0, 1)``, or ``residual=True`` with ``output_dim != input_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int | None = None,
        *,
        residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        out_dim = input_dim if output_dim is None else int(output_dim)
        if out_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {out_dim}")
        if residual and out_dim != input_dim:
            raise ValueError(
                f"residual=True requires output_dim == input_dim; "
                f"got input_dim={input_dim}, output_dim={out_dim}"
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(out_dim)
        self.residual = bool(residual)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        if residual:
            # Zero-init the output projection so f_ctx(x) == x at step 0.
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x_tilde: Tensor) -> Tensor:
        r"""Compute :math:`x_n = f_{\text{ctx}}(\tilde x_n)`.

        Parameters
        ----------
        x_tilde : Tensor of shape (..., input_dim)
            Raw context, typically ``(B, input_dim)`` from
            :meth:`shotcloud.data.ContextEncoder.transform`.

        Returns
        -------
        Tensor of shape (..., output_dim)
            Learned context. In residual mode this equals ``x_tilde``
            exactly at initialization.

        Raises
        ------
        ValueError
            If the last dimension of ``x_tilde`` is not ``input_dim``.
        """
        if x_tilde.shape[-1] != self.input_dim:
            raise ValueError(
                f"x_tilde has last-dim {x_tilde.shape[-1]}, expected input_dim={self.input_dim}"
            )
        h: Tensor = self.fc2(self.drop(self.act(self.fc1(x_tilde))))
        if self.residual:
            return x_tilde + h
        return h
