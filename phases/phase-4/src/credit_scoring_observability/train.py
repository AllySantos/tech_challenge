"""Treina o modelo baseline de crédito, mas só se o contrato aprovar os dados."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from credit_scoring_observability.contracts import validate_batch
from credit_scoring_observability.data import TARGET
from credit_scoring_observability.prep import REFERENCE_CSV

MODEL_PATH = Path("models/baseline.joblib")
METRICS_PATH = Path("reports/baseline_metrics.json")
# id não tem poder preditivo; sex fica de fora por decisão de governança
# (continua nos dados para auditoria de viés, mas não decide o crédito).
EXCLUDED = ["id", "sex", TARGET]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED]


def main() -> None:
    df = pd.read_csv(REFERENCE_CSV)
    if not validate_batch(df):
        sys.exit("Treino abortado: dados reprovados no contrato.")

    features = feature_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "avg_precision": round(average_precision_score(y_test, proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": features,
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, proba >= 0.5, digits=3))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
