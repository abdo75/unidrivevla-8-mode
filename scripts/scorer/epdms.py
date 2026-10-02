"""Reconstruct NAVSIM EPDMS from its sub-scores.

Verified against the per-candidate oracle CSVs on navhard_two_stage: the final
``score`` column equals (product of the multiplicative compliance terms) times a
weighted average of the remaining terms. Each token is scored in a single stage
(the CSV fills either the ``*_stage_one`` or ``*_stage_two`` columns), so the same
formula applies per token to whichever stage's sub-scores are present.

Weights are NAVSIM's canonical PDM weighting (ego-progress 5, time-to-collision 5,
comfort 2, lane-keeping 2, extended-comfort 2); confirmed by reproducing the CSV
``score`` to 1e-3 (see test_epdms.py).
"""

# Multiplicative compliance/safety terms: any zero zeroes the whole score.
MULTIPLICATIVE = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
]

# Weighted-average terms and their weights.
WEIGHTED = {
    "ego_progress": 5.0,
    "time_to_collision_within_bound": 5.0,
    "history_comfort": 2.0,
    "lane_keeping": 2.0,
    "two_frame_extended_comfort": 2.0,
}

# Canonical ordering used for the scorer's prediction vector.
SUBSCORE_KEYS = MULTIPLICATIVE + list(WEIGHTED)


def reconstruct_epdms(subscores):
    """subscores: dict mapping each name in SUBSCORE_KEYS to a value in [0, 1]."""
    penalty = 1.0
    for k in MULTIPLICATIVE:
        penalty *= float(subscores[k])
    denom = sum(WEIGHTED.values())
    numer = sum(w * float(subscores[k]) for k, w in WEIGHTED.items())
    return penalty * (numer / denom)