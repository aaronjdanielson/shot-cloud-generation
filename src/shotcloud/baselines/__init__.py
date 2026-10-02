"""Causal non-neural baselines for shot-cloud generation.

The mainline AC-KDE (paper §2) is a learned context-adaptive density. To
ground its improvement claims we compare against a simple, transparent,
basketball-natural baseline: a per-player grid KDE with shrinkage to
the player's position prior, trained on shots before the prediction
date and evaluated on the same player-games as the AC-KDE.
"""

from shotcloud.baselines.causal_kde import CausalGridKDEBaseline

__all__ = ["CausalGridKDEBaseline"]
