"""Metrics and reporting."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


@dataclass
class Result:
    model: str
    protocol: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    mcc: float
    fit_s: float
    predict_ms_per_sample: float
    n_train: int
    n_test: int
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray | None = None

    def row(self) -> dict:
        d = {
            "Model": self.model,
            "Protocol": self.protocol,
            "Accuracy": round(self.accuracy * 100, 2),
            "Balanced Acc": round(self.balanced_accuracy * 100, 2),
            "F1 Macro": round(self.f1_macro * 100, 2),
            "MCC": round(self.mcc, 4),
            "Fit (s)": round(self.fit_s, 2),
            "Infer (ms)": round(self.predict_ms_per_sample, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
        }
        d.update({f"F1[{k}]": round(v * 100, 2) for k, v in self.per_class_f1.items()})
        return d


def evaluate(model, X_tr, y_tr, X_te, y_te, class_names, model_name, protocol) -> Result:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_te)
    infer_ms = (time.perf_counter() - t1) / max(1, len(X_te)) * 1000

    present = sorted(set(np.unique(y_te)) | set(np.unique(y_pred)))
    f1s = f1_score(y_te, y_pred, average=None, labels=present, zero_division=0)

    return Result(
        model=model_name,
        protocol=protocol,
        accuracy=accuracy_score(y_te, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_te, y_pred),
        f1_macro=f1_score(y_te, y_pred, average="macro", zero_division=0),
        mcc=matthews_corrcoef(y_te, y_pred),
        fit_s=fit_s,
        predict_ms_per_sample=infer_ms,
        n_train=len(X_tr),
        n_test=len(X_te),
        per_class_f1={class_names[c]: float(v) for c, v in zip(present, f1s, strict=True)},
        confusion=confusion_matrix(y_te, y_pred, labels=list(range(len(class_names)))),
    )


def results_frame(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame([r.row() for r in results])


def text_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, target_names=class_names,
                                 digits=4, zero_division=0)
