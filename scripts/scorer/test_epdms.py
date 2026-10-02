"""Verify reconstruct_epdms against real rows from the navhard oracle cand0 CSV.

Run (no pytest needed):  python3 test_epdms.py
"""
from epdms import reconstruct_epdms

# (sub-scores, expected CSV `score`) from real rows of
# uni_oracle_navhard_two_stage_cand0/.../*.csv (stage_one columns).
CASES = [
    # row 0: drivable_area_compliance = 0 -> multiplicative term zeroes the score.
    ({"no_at_fault_collisions": 1.0, "drivable_area_compliance": 0.0,
      "driving_direction_compliance": 0.5, "traffic_light_compliance": 1.0,
      "ego_progress": 1.0, "time_to_collision_within_bound": 1.0,
      "history_comfort": 1.0, "lane_keeping": 1.0,
      "two_frame_extended_comfort": 1.0}, 0.0),
    # row 1: drivable_area_compliance = 0 -> 0.
    ({"no_at_fault_collisions": 1.0, "drivable_area_compliance": 0.0,
      "driving_direction_compliance": 0.5, "traffic_light_compliance": 1.0,
      "ego_progress": 0.8776753319805767, "time_to_collision_within_bound": 1.0,
      "history_comfort": 1.0, "lane_keeping": 1.0,
      "two_frame_extended_comfort": 1.0}, 0.0),
    # row 2: full compliance, EP=0.4562, ext_comfort=0 -> 0.7050514330663533.
    ({"no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0,
      "driving_direction_compliance": 1.0, "traffic_light_compliance": 1.0,
      "ego_progress": 0.4561645858123305, "time_to_collision_within_bound": 1.0,
      "history_comfort": 1.0, "lane_keeping": 1.0,
      "two_frame_extended_comfort": 0.0}, 0.7050514330663533),
]


def main():
    for sub, expected in CASES:
        got = reconstruct_epdms(sub)
        assert abs(got - expected) < 1e-3, "reconstruct=%r expected=%r" % (got, expected)
    print("OK: reconstruct_epdms matches the CSV score on %d real rows" % len(CASES))


if __name__ == "__main__":
    main()