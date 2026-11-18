"""Dictionary-style trip loader used by the archived pipelines."""
import numpy as np  # noqa: F401
from pathlib import Path  # noqa: F401

from src.config import RAW_DIR
from src.data.loader import load_trip


def load_trip_data(trip_id: str, split: str = "train", raw_dir=RAW_DIR) -> dict:
    """Dictionary-based loader compatible with candidate extraction and map pipelines."""
    t = load_trip(split, trip_id, raw_dir=raw_dir)
    return {
        "nodes": t.nodes[["lon", "lat"]].to_numpy(),
        "arcs": t.arcs.to_numpy(),
        "track": t.track[["lon", "lat", "t"]].to_numpy(),
        "route": t.route
    }
