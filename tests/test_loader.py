import pytest

from src.config import RAW_DIR
from src.data.loader import list_trips, load_trip

pytestmark = pytest.mark.skipif(not (RAW_DIR / "train").exists(), reason="raw data not present")


def test_load_first_train_trip():
    tid = list_trips("train")[0]
    trip = load_trip("train", tid)
    assert len(trip.track) > 0
    assert trip.arcs.to_numpy().max() < len(trip.nodes)
    assert trip.route.max() < len(trip.arcs)
