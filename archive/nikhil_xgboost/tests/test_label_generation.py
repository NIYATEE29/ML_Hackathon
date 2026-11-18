
import pandas as pd
import numpy as np

from archive.nikhil_xgboost.candidate_generation import generate_candidates
from archive.nikhil_xgboost.label_generation import generate_labels


class FakeTrip:
    pass


def test_label_generation():
    trip = FakeTrip()

    # Two parallel 100-metre roads.
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
        "y": [2.0, 18.0],
        "t": [0.0, 1.0],
    })

    # Ground-truth route: first road, then second road.
    trip.route = np.array([0, 1])

    candidates = generate_candidates(
        trip,
        radius_m=40.0,
        sample_spacing_m=10.0,
    )

    labeled = generate_labels(trip, candidates)

    # Point 0 is closest to road 0.
    point0 = labeled[labeled["point_idx"] == 0]
    assert point0.loc[point0["arc_id"] == 0, "label"].iloc[0] == 1
    assert point0.loc[point0["arc_id"] == 1, "label"].iloc[0] == 0

    # Point 1 is closest to road 1.
    point1 = labeled[labeled["point_idx"] == 1]
    assert point1.loc[point1["arc_id"] == 1, "label"].iloc[0] == 1
    assert point1.loc[point1["arc_id"] == 0, "label"].iloc[0] == 0

    # Both observations are close enough to the reference route.
    assert labeled["valid_label"].all()
