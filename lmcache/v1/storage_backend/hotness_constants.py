# SPDX-License-Identifier: Apache-2.0
"""
Shared constants for hotness-based cache scoring.

Both the global ``HotnessPolicy`` (cross-tier) and the backend-local
``HotnessCachePolicy`` use the same scoring formula.  All tuning knobs
live here so that the two implementations stay in sync.
"""

HIT_CAP = 32

PREFIX_DECAY = 16.0
AGE_DECAY = 32.0

PREFIX_WEIGHT = 0.45
AGE_WEIGHT = 0.35
HIT_WEIGHT = 0.20

PROMOTION_MARGIN = 0.05
