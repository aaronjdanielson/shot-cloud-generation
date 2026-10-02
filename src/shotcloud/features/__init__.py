"""Causal feature pipelines built outside the model layer.

This package contains pure data-layer feature builders that produce
per-(entity, snapshot) tensors consumed by the model modules in
:mod:`shotcloud.models`:

* :mod:`shotcloud.features.defense_features` — opponent allowed-shot
  team, zone, and reliability features for the opponent reweighting
  term ``D``.
* :mod:`shotcloud.features.matchup_features` — residualized matchup
  features: how players of a similar type shift their zone mix against
  each opponent.
* :mod:`shotcloud.features.usage_features` — expected-usage features
  for the usage residual branch.
"""
