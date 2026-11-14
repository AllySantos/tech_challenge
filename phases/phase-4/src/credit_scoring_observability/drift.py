"""Detecta drift (Evidently: PSI e KS), gera relatórios HTML e mede o desempenho."""

import json
from pathlib import Path

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import recall_score, roc_auc_score

from credit_scoring_observability.data import TARGET
from credit_scoring_observability.prep import REFERENCE_CSV
from credit_scoring_observability.simulate_drift import PERIODS, period_path
from credit_scoring_observability.train import MODEL_PATH, feature_columns

REPORTS_DIR = Path("reports")
PREDICTIONS_CSV = Path("data/production/predictions.csv")
SUMMARY_JSON = REPORTS_DIR / "drift_summary.json"

PSI_THRESHOLD = 0.2  # < 0.1 estável | 0.1 a 0.2 atenção | >= 0.2 drift
KS_ALPHA = 0.05  # o KS devolve p-valor: p < 0.05 => distribuições diferentes
KS_HTML_PERIOD = 4  # relatório KS em HTML só do último período (cada HTML pesa ~4 MB)


def run_drift(reference, current, features, method, threshold):
    """Roda o Evidently e devolve (relatório, {coluna: score})."""
    dd = DataDefinition(numerical_columns=features)
    evaluation = Report([DataDriftPreset(method=method, threshold=threshold)]).run(
        Dataset.from_pandas(current[features], data_definition=dd),
        Dataset.from_pandas(reference[features], data_definition=dd),
    )
    scores = {
        m["config"]["column"]: m["value"]
        for m in evaluation.dict()["metrics"]
        if m["config"]["type"].endswith("ValueDrift")
    }
    return evaluation, scores


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    reference = pd.read_csv(REFERENCE_CSV)
    features = feature_columns(reference)
    model = joblib.load(MODEL_PATH)

    summary, predictions = [], []
    for k in PERIODS:
        current = pd.read_csv(period_path(k))

        # 1) Predições do baseline neste período
        score = model.predict_proba(current[features])[:, 1]
        predictions.append(
            pd.DataFrame(
                {
                    "period": k,
                    "id": current["id"],
                    "score": score,
                    "prediction": (score >= 0.5).astype(int),
                    TARGET: current[TARGET],
                }
            )
        )

        # 2) Data drift: referência x período (PSI em todos; KS para comparar)
        ev_psi, psi = run_drift(reference, current, features, "psi", PSI_THRESHOLD)
        ev_ks, ks = run_drift(reference, current, features, "ks", KS_ALPHA)
        ev_psi.save_html(str(REPORTS_DIR / f"drift_period_{k}_psi.html"))
        if k == KS_HTML_PERIOD:
            ev_ks.save_html(str(REPORTS_DIR / f"drift_period_{k}_ks.html"))

        # 3) Desempenho: é aqui que o concept drift aparece
        summary.append(
            {
                "period": k,
                "psi": {c: round(v, 4) for c, v in psi.items()},
                "ks_pvalue": {c: float(f"{v:.3g}") for c, v in ks.items()},
                "drifted_psi": [c for c, v in psi.items() if v >= PSI_THRESHOLD],
                "drifted_ks": [c for c, v in ks.items() if v < KS_ALPHA],
                "default_rate": round(float(current[TARGET].mean()), 4),
                "auc": round(float(roc_auc_score(current[TARGET], score)), 4),
                "recall": round(float(recall_score(current[TARGET], score >= 0.5)), 4),
            }
        )

    pd.concat(predictions).to_csv(PREDICTIONS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    psi_table = pd.DataFrame({f"periodo_{s['period']}": s["psi"] for s in summary})
    print(f"\nPSI por variável (>= {PSI_THRESHOLD} = drift):")
    print(psi_table.round(3).to_string())
    print()
    for s in summary:
        print(f"periodo {s['period']}: PSI aponta {s['drifted_psi']}")
        print(f"           KS aponta {len(s['drifted_ks'])}/{len(features)} variáveis")


if __name__ == "__main__":
    main()
