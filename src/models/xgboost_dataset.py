
"""Build the supervised candidate dataset for XGBoost using training trips only."""

import pandas as pd

from src.data.loader import list_trips, load_trip
from src.features.candidate_generation import generate_candidates
from src.features.feature_extraction import extract_features
from src.features.label_generation import generate_labels


FEATURE_COLUMNS = [
    "distance",
    "projection_fraction",
    "road_length",
    "road_bearing",
    "gps_heading",
    "heading_difference",
    "speed",
    "time_gap",
    "candidate_rank",
]


def build_training_dataset():
    """
    Build candidate-level X, y, groups, and point IDs from training trips.

    Returns
    -------
    X : DataFrame
        Candidate feature matrix.
    y : Series
        Binary candidate labels (1 = correct, 0 = incorrect).
    groups : Series
        Trip ID for each candidate, used for group-aware CV.
    point_ids : Series
        GPS point index for each candidate, used for Top-1 accuracy.
    """

    X_parts = []
    y_parts = []
    group_parts = []
    point_parts = []

    # IMPORTANT: We only load the training partition.
    trip_ids = list_trips("train")

    if not trip_ids:
        raise RuntimeError(
            "No training trips found in data/raw/train. "
            "Place the PathSnap training trip folders there first."
        )

    print(f"Found {len(trip_ids)} training trips.")

    for trip_id in trip_ids:
        print(f"\nProcessing training trip: {trip_id}")

        trip = load_trip("train", trip_id)

        # Step 1: Generate possible road candidates for GPS points.
        candidates = generate_candidates(trip)

        if candidates.empty:
            print("  Skipped: no candidates generated.")
            continue

        # Step 2: Extract numerical features for each candidate.
        features = extract_features(trip, candidates)

        if features.empty:
            print("  Skipped: no features generated.")
            continue

        # Step 3: Generate candidate labels from the reference route.
        labeled = generate_labels(trip, features)

        # Step 4: Keep only observations with reliable labels.
        labeled = labeled[labeled["valid_label"]].copy()

        if labeled.empty:
            print("  Skipped: no reliable labeled observations.")
            continue

        # Step 5: Keep only points having at least one positive candidate.
        positive_points = labeled.loc[
            labeled["label"] == 1, "point_idx"
        ].unique()

        labeled = labeled[
            labeled["point_idx"].isin(positive_points)
        ].copy()

        if labeled.empty:
            print("  Skipped: no positive candidates.")
            continue

        # Verify all required features are available.
        missing_features = [
            col for col in FEATURE_COLUMNS
            if col not in labeled.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing features in trip {trip_id}: {missing_features}"
            )

        # Step 6: Store features, labels, trip IDs and GPS point IDs.
        X_parts.append(labeled[FEATURE_COLUMNS].copy())
        y_parts.append(labeled["label"].astype(int).copy())
        group_parts.append(
            pd.Series(trip_id, index=labeled.index, dtype=str)
        )
        point_parts.append(
            labeled["point_idx"].astype(int).copy()
        )

        print(f"  Retained candidate rows: {len(labeled):,}")
        print(f"  Retained GPS points: {labeled['point_idx'].nunique():,}")
        print(f"  Positive candidates: {int(labeled['label'].sum()):,}")

    # Ensure that at least one trip produced usable training data.
    if not X_parts:
        raise RuntimeError(
            "No training data could be assembled from the available trips."
        )

    # Combine all trips into one dataset.
    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True)
    point_ids = pd.concat(point_parts, ignore_index=True)

    # Basic summary.
    print("\n========== TRAINING DATASET SUMMARY ==========")
    print(f"Total candidate rows: {len(X):,}")
    print(f"Total GPS points: {point_ids.nunique():,}")
    print(f"Total trips: {groups.nunique():,}")
    print(f"Positive labels: {int(y.sum()):,}")
    print(f"Negative labels: {int((y == 0).sum()):,}")
    print(f"Positive label ratio: {y.mean():.4f}")

    return X, y, groups, point_ids
