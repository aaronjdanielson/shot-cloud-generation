"""Shared pytest fixtures for the shotcloud test suite.

An autouse fixture reseeds every random number generator before each test so
results are deterministic. matplotlib's non-interactive Agg backend is selected
before any other matplotlib import so plotting tests run headless.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset Python, NumPy, and (if available) PyTorch RNGs to seed 42 before each test."""
    random.seed(42)
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass
