"""Causal non-neural baselines for shot-cloud generation.

:class:`CausalGridKDEBaseline` is a per-player grid KDE shrunk toward the
player's position density, fit only on shots before a training cutoff.
It serves as a simple, transparent reference for the AC-KDE and is
scored on the same player-games with the same metrics.
"""

from shotcloud.baselines.causal_kde import CausalGridKDEBaseline

__all__ = ["CausalGridKDEBaseline"]
