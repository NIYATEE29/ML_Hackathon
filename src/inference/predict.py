"""Inference path: raw trip files -> scored candidates (-> decoded route).

Nothing here reads the .route file or any label, so the API can call it on a hidden
evaluation trip. Ground truth is only ever touched by the scorer on the server side.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RAW_DIR
from src.data.loader import Trip, load_trip
from src.features.candidates import FEATURES, build_trip_table


def load_model(name: str, models_dir: Path = MODELS_DIR):
    """Load a saved model bundle (dict with 'model' and 'features')."""
    bundle = joblib.load(models_dir / f"{name}.joblib")
    assert bundle["features"] == FEATURES, "saved model was trained with a different feature list"
    return bundle["model"]


def score_trip(trip: Trip, model) -> pd.DataFrame:
    """Candidate rows for one trip with column `p` = P(candidate is the travelled arc)."""
    table, _ = build_trip_table(trip, with_labels=False)
    table["p"] = model.predict_proba(table[FEATURES].to_numpy(np.float32))[:, 1] if len(table) else []
    return table


def load_trip_for_inference(partition: str, trip_id: str, raw_dir: Path = RAW_DIR) -> Trip:
    return load_trip(partition, trip_id, raw_dir=raw_dir, with_route=False)
