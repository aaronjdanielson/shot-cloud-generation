"""Global seeding and determinism flags for reproducible runs.

:func:`set_global_determinism` seeds every random-number generator the
package uses and sets the PyTorch flags required for run-to-run
reproducibility:

* ``torch.use_deterministic_algorithms(True)`` -- selects deterministic
  implementations where one exists.
* ``torch.backends.cudnn.deterministic = True`` and
  ``torch.backends.cudnn.benchmark = False`` -- disable cuDNN's autotuned
  kernel selection on CUDA.
* ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` -- required for cuBLAS determinism.
* ``PYTHONHASHSEED`` -- fixes hash-order-dependent computations.

Without these flags, two runs of the same script on the same machine can
differ in the fourth to sixth decimal place.

Several MPS ops (softmax, ``scatter_add``, reductions over the last
dimension) have no bit-deterministic backend, so
``torch.use_deterministic_algorithms(True)`` would raise on them. The
default ``warn_only=True`` makes MPS runs emit a one-time warning instead
of failing, while still selecting deterministic kernels where available.

Examples
--------
Call it at the top of a script's ``main()``::

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
    """Seed all random-number generators and set PyTorch determinism flags.

    Call once from a script's ``main()``, before any tensor construction
    or DataLoader instantiation. Idempotent: repeated calls with the same
    seed leave the same state.

    Parameters
    ----------
    seed : int
        The master seed. Plumbed to Python's ``random``, NumPy, PyTorch
        (CPU + CUDA + MPS), and ``PYTHONHASHSEED``.
    device : {"cpu", "cuda", "mps"} or str
        Target device. Used to decide which device-specific flags to
        set. An unrecognized string is treated like ``"cpu"``: only the
        device-agnostic flags are set (the cuDNN flags are also set
        whenever CUDA is available).
    warn_only_on_nondeterministic_ops : bool, default True
        Forward to :func:`torch.use_deterministic_algorithms`. ``True``
        (the default) is required for MPS, where several ops lack
        deterministic implementations; PyTorch then emits a one-time
        warning instead of raising. Set ``False`` for strict CPU/CUDA
        runs when you want a hard error on any non-deterministic op.

    Notes
    -----
    MPS is not bit-deterministic for several ops. For bit-exact
    reproducibility, train and evaluate on CPU or CUDA; MPS runs agree to
    roughly six decimal places.
    """
    # Python + NumPy + hash-order
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # The global NumPy RNG is seeded for callees that draw from it rather
    # than from ``np.random.default_rng`` (e.g. AdaptiveKDE's max_history
    # sampler).
    np.random.seed(seed)  # noqa: NPY002

    # torch.manual_seed also seeds CUDA and MPS; the explicit setters are
    # called as well for robustness across PyTorch versions.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and getattr(torch.mps, "manual_seed", None) is not None:
        # The call can raise on builds without a usable MPS backend.
        with contextlib.suppress(Exception):
            torch.mps.manual_seed(seed)

    # warn_only=True is required on MPS, where scatter_add and some
    # reductions have no deterministic implementation. PyTorch < 1.8 lacks
    # this API, in which case only the seeds above apply.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=warn_only_on_nondeterministic_ops)

    # CUDA-specific flags and workspace config.
    if device == "cuda" or torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


__all__ = ["set_global_determinism"]
