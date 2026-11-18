
"""XGBoost candidate-scoring model for PathSnap."""

from xgboost import XGBClassifier

from src.config import SEED


class XGBoostCandidateModel:
    """Predict the probability that a candidate road is the correct match."""

    def __init__(self, **params):
        default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": SEED,
            "tree_method": "hist",
            "n_jobs": -1,
        }

        # Allow custom parameters to override the baseline.
        default_params.update(params)

        self.model = XGBClassifier(**default_params)

    def fit(self, X, y):
        """Train the model on candidate features and binary labels."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Return the probability that each candidate is the correct road."""
        return self.model.predict_proba(X)
