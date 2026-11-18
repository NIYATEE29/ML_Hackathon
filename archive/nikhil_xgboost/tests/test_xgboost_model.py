import numpy as np
import pytest

pytest.importorskip("xgboost")
import pandas as pd

from archive.nikhil_xgboost.xgboost_model import XGBoostCandidateModel


def test_xgboost_smoke_test():
    """Verify XGBoost can train and return valid probabilities."""

    # Create a small synthetic dataset (not PathSnap data)
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "distance": rng.uniform(0, 40, 100),
        "heading_difference": rng.uniform(0, 180, 100),
        "speed": rng.uniform(0, 20, 100),
        "road_length": rng.uniform(5, 500, 100),
    })

    # Synthetic binary labels for software testing
    y = (X["distance"] < 15).astype(int)

    model = XGBoostCandidateModel(
        n_estimators=20,
        max_depth=3,
    )

    # Fit should complete without errors
    returned_model = model.fit(X, y)

    # fit() should return the model wrapper itself
    assert returned_model is model

    # Generate probabilities
    probabilities = model.predict_proba(X)

    # Check output dimensions
    assert probabilities.shape == (100, 2)

    # Probabilities must be valid
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    # Each row's probabilities must sum to 1
    assert np.allclose(probabilities.sum(axis=1), 1.0)