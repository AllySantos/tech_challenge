import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
print(f"SRC_DIR: {SRC_DIR}")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----------------------------------------------------------- #


import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from services.mlflow_service import MLFlowService
from training.train import preprocessing
from utils.loaders import load_model

mlflow_service = MLFlowService(experiment_name="churn-prediction-baseline-comparison")

RANDOM_SEED = 42
COST_FP = 50  # custo de uma campanha de retenção desnecessária (R$)
COST_FN = 500  # receita perdida quando um churner não é detectado (R$)
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "custo_total": int(fp * COST_FP + fn * COST_FN),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }


def print_metrics(name: str, m: dict):
    print(f"\n{'=' * 52}")
    print(f"  {name}")
    print(f"{'=' * 52}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:<12}: {m[k]:.4f}")
    print(f"  custo_total: R$ {m['custo_total']:,}  (FP={m['fp']}, FN={m['fn']})")


def evaluate_baseline(name, clf, X_train, y_train, X_test, y_test, params=None):
    clf.fit(X_train, y_train)
    proba = (
        clf.predict_proba(X_test)[:, 1]
        if hasattr(clf, "predict_proba")
        else clf.decision_function(X_test)
    )
    m = compute_metrics(y_test, proba)

    with mlflow_service.start_run(run_name=name):
        if params:
            mlflow_service.log_params(params)
        mlflow_service.log_params({"model_type": name})
        mlflow_service.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow_service.log_sklearn_model(clf, "model")

    print_metrics(name, m)
    return m, proba


def main():
    # 1. Dados (reaproveitamento do pré-processamento do treinamento)
    X_train, y_train, X_test, y_test = preprocessing()

    # 2. Baselines Clássicos
    baselines = [
        (
            "Logistic Regression",
            LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED),
            {"C": 0.1, "penalty": "l2"},
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=5, random_state=RANDOM_SEED),
            {"max_depth": 5},
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_SEED
            ),
            {"n_estimators": 100, "max_depth": 8},
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_SEED
            ),
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        ),
    ]

    # 3. Resultados dos baselines
    results = {}
    for name, clf, params in baselines:
        print(name)
        results[name], _ = evaluate_baseline(name, clf, X_train, y_train, X_test, y_test, params)

    # 4. MLP
    input_dim = X_train.shape[1]
    model = load_model(input_dim=input_dim, checkpoint_name="model.pth")
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(DEVICE)
        mlp_proba = model(tensor).cpu().numpy()

    mlp_metrics = compute_metrics(y_test, mlp_proba)
    with mlflow_service.start_run(run_name="MLP_evaluate"):
        mlflow_service.log_metrics(
            {k: v for k, v in mlp_metrics.items() if isinstance(v, (int, float))}
        )

    results["MLP PyTorch"] = mlp_metrics
    print_metrics("MLP PyTorch", mlp_metrics)

    # 5. Tabela comparativa
    summary = (
        pd.DataFrame(results)
        .T[["accuracy", "precision", "recall", "f1", "roc_auc", "custo_total"]]
        .sort_values("roc_auc", ascending=False)
    )
    print(f"\n\n{'=' * 60}")
    print("  COMPARAÇÃO FINAL")
    print(f"{'=' * 60}")
    print(summary.round(4).to_string())

    # 5. Análise de threshold — MLP
    print(f"\n\n{'=' * 60}")
    print("  TRADE-OFF DE CUSTO (MLP) — variando threshold")
    print(f"  COST_FP=R${COST_FP} | COST_FN=R${COST_FN}")
    print(f"{'=' * 60}")
    rows = []
    for t in np.arange(0.1, 0.9, 0.05):
        m = compute_metrics(y_test, mlp_proba, threshold=t)
        rows.append(
            {
                "threshold": round(t, 2),
                "recall": m["recall"],
                "precision": m["precision"],
                "custo": m["custo_total"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    df_thresh = pd.DataFrame(rows)
    best = df_thresh.loc[df_thresh["custo"].idxmin()]
    print(df_thresh.to_string(index=False))
    print(
        f"\n✅ Threshold ótimo: {best['threshold']:.2f}  "
        f"Recall={best['recall']:.3f}  Custo=R${best['custo']:,}"
    )


if __name__ == "__main__":
    main()
