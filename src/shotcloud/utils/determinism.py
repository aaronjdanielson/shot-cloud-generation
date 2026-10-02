"""Global determinism helpers (audit fix H5, 2026-06-10).

The pre-H5 codebase seeded ``torch`` and ``numpy`` inside each entry-point
``main()`` but never set the determinism flags PyTorch needs for cross-
machine, cross-run bit-reproducibility:

* ``torch.use_deterministic_algorithms(True)`` — forces PyTorch to use
  deterministic implementations where one exists (and raise on ops that
  don't, unless ``warn_only=True``).
* ``torch.backends.cudnn.deterministic = True`` + ``.benchmark = False``
  — disables cuDNN's autotuned kernel selection on CUDA.
* ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` — required for cuBLAS determinism
  on PyTorch ≥1.8.
* ``PYTHONHASHSEED`` — needed for some hash-order-dependent
  computations (e.g., dict iteration with hash-keyed inputs).

Without these, two runs of the same script on the same machine can
differ at the 4th–6th decimal — small enough to miss in casual
inspection, large enough to fail strict reproducibility for an AOAS
submission.

**MPS caveat.** As of PyTorch 2.x, several MPS ops (softmax,
scatter_add, reduce-on-last-dim, …) do not have bit-deterministic
backends and ``torch.use_deterministic_algorithms(True)`` will raise on
them. We pass ``warn_only=True`` so MPS runs emit a one-time warning
instead of crashing, while still flipping the flag for the
deterministic-when-available subset.

Usage from a script's ``main()``::

    from shotcloud.utils.determinism import set_global_determinism

    def main():
        args = _parse_args()
        set_global_determinism(args.seed, device=args.device)
        ...
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Literal

import numpy as np
import torch


def set_global_determinism(
    seed: int,
    *,
    device: Literal["cpu", "cuda", "mps"] | str = "cpu",
    warn_only_on_nondeterministic_ops: bool = True,
) -> None:
    """Seed every RNG and flip every determinism flag PyTorch supports.

    Calls this helper from every script's ``main()`` BEFORE any tensor
    construction or DataLoader instantiation. Idempotent — safe to call
    multiple times with the same seed.

    Parameters
    ----------
    seed : int
        The master seed. Plumbed to Python's ``random``, NumPy, PyTorch
        (CPU + CUDA + MPS), and ``PYTHONHASHSEED``.
    device : {"cpu", "cuda", "mps"} or str
        Target device. Used to decide which device-specific flags to
        flip. Passing an unknown string is treated like "cpu" (only the
        device-agnostic flags fire).
    warn_only_on_nondeterministic_ops : bool, default True
        Forward to :func:`torch.use_deterministic_algorithms`. ``True``
        (the default) is required for MPS, where several ops lack
        deterministic implementations; PyTorch then emits a one-time
        warning instead of raising. Set ``False`` for strict CPU/CUDA
        runs when you want a hard error on any non-deterministic op.

    Notes
    -----
    PyTorch MPS is NOT bit-deterministic for several ops as of
    PyTorch 2.x. For strict bit reproducibility, train + evaluate on
    CPU or CUDA. MPS runs in this codebase are reproducible up to ~6
    decimal places, which is sufficient for paper-level claims but not
    for strict regression testing.
    """
    # Python + NumPy + hash-order
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # The legacy global numpy RNG is seeded for back-compat with
    # callees that haven't migrated to ``np.random.default_rng``
    # (e.g. AdaptiveKDE's max_history sampler).
    np.random.seed(seed)  # noqa: NPY002

    # PyTorch base RNG (CPU). torch.manual_seed also seeds CUDA + MPS
    # under the hood; we call the explicit setters as well for
    # robustness on older versions.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and getattr(torch.mps, "manual_seed", None) is not None:
        # MPS has its own manual_seed since PyTorch 2.x. Older MPS
        # branches lack the call — suppress and continue.
        with contextlib.suppress(Exception):
            torch.mps.manual_seed(seed)

    # Deterministic-algorithms flag. warn_only=True is REQUIRED for MPS
    # paths (scatter_add and a few reductions don't have deterministic
    # implementations as of PyTorch 2.x). Older PyTorch (<1.8) lacks
    # this API; the seed-only behavior is the best we can do then.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=warn_only_on_nondeterministic_ops)

    # CUDA-specific flags + workspace config.
    if device == "cuda" or torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for cuBLAS determinism on PyTorch >= 1.8.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


__all__ = ["set_global_determinism"]
