
import pandas as pd

from archive.nikhil_xgboost.candidate_generation import generate_candidates


class FakeTrip:
    pass


def test_candidate_generation():
    trip = FakeTrip()

    # Two horizontal roads, 20 metres apart.
    trip.nodes = pd.DataFrame({
        "x": [0.0, 100.0, 0.0, 100.0],
        "y": [0.0, 0.0, 20.0, 20.0],
    })

    trip.arcs = pd.DataFrame({
        "from_node": [0, 2],
        "to_node": [1, 3],
    })

    trip.track = pd.DataFrame({
        "x": [50.0, 50.0],
        "y": [3.0, 80.0],
    })

    candidates = generate_candidates(
        trip,
        radius_m=40.0,
        sample_spacing_m=10.0,
    )

    # First GPS observation should find both nearby roads.
    first_point_candidates = candidates[
        candidates["point_idx"] == 0
    ]

    assert set(first_point_candidates["arc_id"]) == {0, 1}

    # The second observation is far from both roads.
    second_point_candidates = candidates[
        candidates["point_idx"] == 1
    ]

    assert second_point_candidates.empty
