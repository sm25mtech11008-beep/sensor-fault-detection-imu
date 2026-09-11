"""Model zoo.

Every model is wrapped in a Pipeline with its own scaler, so that scaling is
fit on training folds only. In v1 the scaled and unscaled matrices were built
once outside the CV loop and reused, which quietly leaks test statistics into
every cross-validation fold.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def build_models(seed: int = 42, fast: bool = False) -> dict[str, Pipeline]:
    """Return name -> pipeline.

    ``fast=True`` drops SVM and KNN, whose fit/predict cost is superlinear and
    dominates sweeps over many protocols.
    """
    models: dict[str, Pipeline] = {
        # Baselines first: a result is only meaningful relative to these.
        "Majority Baseline": _scaled(DummyClassifier(strategy="most_frequent")),
        "Decision Stump": _scaled(DecisionTreeClassifier(max_depth=3, random_state=seed)),
        "Logistic Regression": _scaled(
            LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "Random Forest": _scaled(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "HistGradientBoosting": _scaled(
            HistGradientBoostingClassifier(max_iter=300, random_state=seed)
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = _scaled(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=-1,
                random_state=seed,
                verbosity=0,
            )
        )

    if not fast:
        models["KNN (k=7)"] = _scaled(
            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
        )
        models["SVM (RBF)"] = _scaled(
            SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced",
                probability=False, random_state=seed)
        )

    return models
