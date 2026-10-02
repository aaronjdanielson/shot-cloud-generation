"""Smoke tests verifying the package skeleton is importable and intact."""

from __future__ import annotations

import importlib

import pytest

EXPECTED_SUBPACKAGES = [
    "shotcloud.data",
    "shotcloud.grids",
    "shotcloud.kde",
    "shotcloud.models",
    "shotcloud.training",
    "shotcloud.evaluation",
    "shotcloud.viz",
    "shotcloud.simulation",
    "shotcloud.utils",
]


def test_top_level_import() -> None:
    import shotcloud

    assert hasattr(shotcloud, "__version__")
    assert isinstance(shotcloud.__version__, str)
    # SemVer-ish: at least major.minor.patch.
    parts = shotcloud.__version__.split(".")
    assert len(parts) >= 3, f"unexpected version format: {shotcloud.__version__!r}"


@pytest.mark.parametrize("module_name", EXPECTED_SUBPACKAGES)
def test_subpackage_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__name__ == module_name


def test_core_dependencies_importable() -> None:
    """Verify the dependencies declared in pyproject.toml load cleanly."""
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import omegaconf  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import yaml  # noqa: F401
