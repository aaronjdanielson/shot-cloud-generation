"""Tests for :func:`shotcloud.utils.determinism.set_global_determinism`."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from shotcloud.utils.determinism import set_global_determinism


def test_sets_pythonhashseed_env() -> None:
    set_global_determinism(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"


def test_python_random_is_reproducible() -> None:
    set_global_determinism(0)
    a = [random.random() for _ in range(5)]
    set_global_determinism(0)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_numpy_legacy_seed_is_reproducible() -> None:
    set_global_determinism(123)
    a = np.random.rand(10)
    set_global_determinism(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_torch_cpu_seed_is_reproducible() -> None:
    set_global_determinism(7)
    a = torch.randn(20)
    set_global_determinism(7)
    b = torch.randn(20)
    assert torch.equal(a, b)


def test_idempotent_same_seed_same_state() -> None:
    """Repeated calls with the same seed leave the same RNG state."""
    set_global_determinism(99)
    set_global_determinism(99)  # second call same seed
    a = torch.randn(50)
    set_global_determinism(99)
    b = torch.randn(50)
    assert torch.equal(a, b)


def test_different_seeds_give_different_sequences() -> None:
    """Different seeds give different sequences."""
    set_global_determinism(0)
    a = torch.randn(20)
    set_global_determinism(1)
    b = torch.randn(20)
    assert not torch.equal(a, b)


def test_cuda_flags_set_when_cuda_available() -> None:
    """With CUDA available, the cuDNN flags and ``CUBLAS_WORKSPACE_CONFIG`` are set;
    without CUDA, ``device="cuda"`` is accepted without error."""
    if not torch.cuda.is_available():
        # Without CUDA, only check that the call does not raise.
        set_global_determinism(0, device="cuda")
        return
    set_global_determinism(0, device="cuda")
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def test_use_deterministic_algorithms_is_flipped() -> None:
    """Deterministic algorithms are enabled (with ``warn_only=True`` by default), so a
    non-deterministic op produces a warning instead of silently varying."""
    set_global_determinism(0)
    # PyTorch reads back the flag via torch.are_deterministic_algorithms_enabled.
    assert torch.are_deterministic_algorithms_enabled() is True
