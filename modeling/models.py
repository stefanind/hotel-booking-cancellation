"""Small reusable model definitions for the cancellation notebooks."""

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare_data import CANDIDATE_MODEL_FEATURES


CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
]

NUMERIC_FEATURES = [
    column
    for column in CANDIDATE_MODEL_FEATURES
    if column not in CATEGORICAL_FEATURES
]


def make_logistic_pipeline(c_value=1.0):
    """Build the shared one-hot-encoded logistic regression pipeline."""

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def as_catboost_features(features):
    """Return a copy with categorical labels represented as plain strings."""

    converted = features.copy()
    for column in CATEGORICAL_FEATURES:
        converted[column] = converted[column].astype(str)
    return converted


def make_catboost_model(depth=6):
    """Build the shared CatBoost candidate with the fixed common settings."""

    return CatBoostClassifier(
        iterations=500,
        depth=depth,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
