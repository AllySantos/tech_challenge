"""Análise causal por intervenção: quanto da queda de AUC vem de cada causa?

Como o drift é simulado, controlamos o mecanismo: ligamos UMA causa por vez (como um
experimento controlado, o "do-operator") e comparamos com o controle (nenhuma causa).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from credit_scoring_observability.data import TARGET
from credit_scoring_observability.prep import PRODUCTION_BASE_CSV
from credit_scoring_observability.simulate_drift import PERIODS, apply_drift
from credit_scoring_observability.train import MODEL_PATH, feature_columns

REPORT_JSON = Path("reports/causal_analysis.json")
REPORT_MD = Path("reports/causal_analysis.md")
N_REPS = 30  # repetições com sementes diferentes, para não depender de uma sorte só

SEVERE = PERIODS[4]  # intensidade do período mais severo
OFF = dict(inflation=0.0, delay_prob=0.0, younger_years=0, concept_rate=0.0)
DATA_CAUSES = ("inflation", "delay_prob", "younger_years")


def only(*causes: str) -> dict:
    """Liga só as causas pedidas (na intensidade do período 4); o resto fica desligado."""
    return {k: (SEVERE[k] if k in causes else OFF[k]) for k in OFF}


SCENARIOS = {
    "controle (nenhuma causa)": only(),
    "só inflação (faturas)": only("inflation"),
    "só mais atrasos (pay_1)": only("delay_prob"),
    "só perfil mais jovem (age)": only("younger_years"),
    "só data drift (as 3 causas)": only(*DATA_CAUSES),
    "só concept drift": only("concept_rate"),
    "tudo (período 4)": only(*DATA_CAUSES, "concept_rate"),
}


def evaluate(base, model, features, params, seed):
    rng = np.random.default_rng(seed)
    df = apply_drift(base, rng=rng, **params)
    score = model.predict_proba(df[features])[:, 1]
    return (
        roc_auc_score(df[TARGET], score),
        recall_score(df[TARGET], score >= 0.5),
        df[TARGET].mean(),
    )


def main() -> None:
    base = pd.read_csv(PRODUCTION_BASE_CSV)
    model = joblib.load(MODEL_PATH)
    features = feature_columns(base)

    rows = []
    for name, params in SCENARIOS.items():
        res = np.array([evaluate(base, model, features, params, s) for s in range(N_REPS)])
        rows.append({
            "scenario": name,
            "auc": res[:, 0].mean(), "auc_std": res[:, 0].std(),
            "recall": res[:, 1].mean(), "default_rate": res[:, 2].mean(),
        })
    table = pd.DataFrame(rows)
    table["delta_auc"] = table["auc"] - table.loc[0, "auc"]

    d = table.set_index("scenario")["delta_auc"]
    data, concept, total = d["só data drift (as 3 causas)"], d["só concept drift"], d["tudo (período 4)"]
    interaction = total - data - concept
    attribution = {"data_drift": data, "concept_drift": concept, "interaction": interaction, "total": total}

    # Relatório em Markdown (vai para o README / vídeo)
    lines = [
        "| Scenario | AUC (mean ± sd) | ΔAUC vs control | Recall | Default rate |",
        "|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.scenario} | {r.auc:.3f} ± {r.auc_std:.3f} | {r.delta_auc:+.3f} "
                     f"| {r.recall:.3f} | {r.default_rate:.1%} |")
    lines += [
        "",
        f"Attribution of the AUC drop ({N_REPS} repetitions): data drift {data:+.3f}, "
        f"concept drift {concept:+.3f}, interaction {interaction:+.3f}, total {total:+.3f}.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({
        "n_reps": N_REPS, "scenarios": table.round(4).to_dict("records"),
        "attribution": {k: round(float(v), 4) for k, v in attribution.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(table.round(3).to_string(index=False))
    print("\nAtribuição da queda de AUC (período 4):")
    for name, value in attribution.items():
        share = f"{value / total:>6.0%} do total" if total else ""
        print(f"  {name:<14} {value:+.3f}   {share}")


if __name__ == "__main__":
    main()