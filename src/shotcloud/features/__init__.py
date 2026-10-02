"""Causal feature pipelines built outside the model layer.

This package contains pure-data-layer feature builders. They produce
per-(entity, snapshot) tensors that downstream model modules
(:mod:`shotcloud.models.*`) consume as inputs to their relevance
heads.

Currently houses:

* :mod:`shotcloud.features.defense_features` — opponent allowed-shot
  team / zone / reliability features for PR-D1's defensive scorer.
"""
