"""Auditoria de viés: o modelo trata grupos de forma diferente?

Usa o mesmo conjunto de teste do treino. "Flag" = o modelo prevê calote (resultado adverso
para o cliente). Quanto maior a diferença entre grupos, maior o alerta de viés.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_scoring_observability.data import TARGET
from credit_scoring_observability.prep import REFERENCE_CSV
from credit_scoring_observability.train import MODEL_PATH, feature_columns

REPORT_MD = Path("reports/fairness_audit.md")
REPORT_JSON = Path("reports/fairness_audit.json")
THRESHOLD = 0.5
ATTRIBUTES = ["sex_group", "age_group", "marriage_group"]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sex_group"] = df["sex"].map({1: "male", 2: "female"})
    df["age_group"] = pd.cut(
        df["age"], [0, 29, 44, 200], labels=["<30", "30-44", "45+"]
    )
    df["marriage_group"] = df["marriage"].map({1: "married", 2: "single", 3: "other"})
    return df


def group_metrics(df: pd.DataFrame, attr: str) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(attr, observed=True):
        y, pred = g[TARGET], g["pred"]
        rows.append(
            {
                "attribute": attr,
                "group": str(group),
                "n": len(g),
                "actual_default": y.mean(),
                "flag_rate": pred.mean(),
                "tpr": pred[
                    y == 1
                ].mean(),  # dos que de fato deram calote, quantos o modelo pegou
                "fpr": pred[
                    y == 0
                ].mean(),  # dos bons pagadores, quantos o modelo marcou errado
                "auc": roc_auc_score(y, g["score"]) if y.nunique() == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(REFERENCE_CSV)
    features = feature_columns(df)
    # Mesmo split do train.py: mesma semente, mesma estratificação.
    _, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df[TARGET], random_state=42
    )
    test = add_groups(df.loc[test_idx])

    model = joblib.load(MODEL_PATH)
    test["score"] = model.predict_proba(test[features])[:, 1]
    test["pred"] = (test["score"] >= THRESHOLD).astype(int)

    tables, summary = [], {}
    for attr in ATTRIBUTES:
        t = group_metrics(test, attr)
        tables.append(t)
        summary[attr] = {
            "flag_rate_ratio": round(
                float(t["flag_rate"].min() / t["flag_rate"].max()), 3
            ),
            "tpr_gap": round(float(t["tpr"].max() - t["tpr"].min()), 3),
            "fpr_gap": round(float(t["fpr"].max() - t["fpr"].min()), 3),
            "auc_gap": round(float(t["auc"].max() - t["auc"].min()), 3),
        }
    result = pd.concat(tables, ignore_index=True)

    lines = [
        "| Attribute | Group | n | Actual default | Flag rate | TPR | FPR | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result.itertuples():
        lines.append(
            f"| {r.attribute} | {r.group} | {r.n} | {r.actual_default:.1%} | {r.flag_rate:.1%} "
            f"| {r.tpr:.3f} | {r.fpr:.3f} | {r.auc:.3f} |"
        )
    lines += [
        "",
        "| Attribute | Flag-rate ratio (min/max) | TPR gap | FPR gap | AUC gap |",
        "|---|---|---|---|---|",
    ]
    for attr, s in summary.items():
        lines.append(
            f"| {attr} | {s['flag_rate_ratio']} | {s['tpr_gap']} | {s['fpr_gap']} | {s['auc_gap']} |"
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {
                "threshold": THRESHOLD,
                "groups": result.round(4).to_dict("records"),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(result.round(3).to_string(index=False))
    print("\nResumo das diferenças entre grupos:")
    for attr, s in summary.items():
        print(f"  {attr:<15} {s}")


if __name__ == "__main__":
    main()
