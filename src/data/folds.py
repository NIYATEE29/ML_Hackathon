import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import KFold

from src.config import SEED, N_FOLDS, PROCESSED_DATA_DIR
from src.data.loader import list_trips

FOLDS_FILE = PROCESSED_DATA_DIR / "folds.json"

def generate_and_save_folds(n_splits: int = N_FOLDS, seed: int = SEED) -> Dict:
    """
    Generates fixed, reproducible trip-level cross-validation folds on the 80 training trips.
    Each trip remains entirely within either the training or validation fold (zero trip-level leakage).
    """
    train_trips = np.array(list_trips("train"))
    if len(train_trips) == 0:
        raise RuntimeError("No training trips found to create folds.")

    # Shuffle trips deterministically using SEED
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(train_trips))
    shuffled_trips = train_trips[shuffled_indices]

    kf = KFold(n_splits=n_splits, shuffle=False)
    folds_data = {
        "seed": seed,
        "n_splits": n_splits,
        "total_train_trips": len(train_trips),
        "folds": []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(shuffled_trips)):
        fold_train_trips = sorted(shuffled_trips[train_idx].tolist())
        fold_val_trips = sorted(shuffled_trips[val_idx].tolist())
        folds_data["folds"].append({
            "fold": fold_idx,
            "train_count": len(fold_train_trips),
            "val_count": len(fold_val_trips),
            "train_trips": fold_train_trips,
            "val_trips": fold_val_trips
        })

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FOLDS_FILE, "w") as f:
        json.dump(folds_data, f, indent=2)

    print(f"Generated {n_splits} folds across {len(train_trips)} trips. Saved to {FOLDS_FILE}")
    return folds_data

def load_folds() -> Dict:
    """Loads existing folds from JSON, generating them if not yet present."""
    if not FOLDS_FILE.exists():
        return generate_and_save_folds()
    with open(FOLDS_FILE, "r") as f:
        return json.load(f)

def get_fold_trips(fold_idx: int) -> Tuple[List[str], List[str]]:
    """Returns (train_trips, val_trips) for a given fold index."""
    folds_data = load_folds()
    if fold_idx < 0 or fold_idx >= len(folds_data["folds"]):
        raise IndexError(f"Fold index {fold_idx} out of range (0 to {len(folds_data['folds']) - 1})")
    fold = folds_data["folds"][fold_idx]
    return fold["train_trips"], fold["val_trips"]

if __name__ == "__main__":
    generate_and_save_folds()
