"""Datasets and trainers for the marked point-process model.

Provides the per-player-game dataset (:class:`GibbsShotDataset`), the
player and opponent vocabularies, the joint trainer :func:`train_gibbs`,
and the single-factor trainers used to pretrain the count, timing, and
presence heads, each with its training-history record.

The per-shot dataset and trainer for the deprecated grid-cell decoder live
in :mod:`shotcloud.legacy_pivot`.
"""

from shotcloud.training.dataset import OpponentVocab, PlayerVocab
from shotcloud.training.gibbs_dataset import N_TIMING_BINS, GibbsShotDataset, PerGameTable
from shotcloud.training.train_gibbs import (
    CountOnlyTrainHistory,
    GibbsTrainHistory,
    PresenceTrainHistory,
    TimingOnlyTrainHistory,
    train_count_only,
    train_gibbs,
    train_presence_only,
    train_timing_only,
)

__all__ = [
    "N_TIMING_BINS",
    "CountOnlyTrainHistory",
    "GibbsShotDataset",
    "GibbsTrainHistory",
    "OpponentVocab",
    "PerGameTable",
    "PlayerVocab",
    "PresenceTrainHistory",
    "TimingOnlyTrainHistory",
    "train_count_only",
    "train_gibbs",
    "train_presence_only",
    "train_timing_only",
]
