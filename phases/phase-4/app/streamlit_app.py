"""Credit Scoring Observability: interactive replay of the monitoring results.

Reads only pre-computed, aggregated results from ./data (no model, no row-level data).
Alert thresholds mirror src/credit_scoring_observability/pipeline.py.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"

# Same rules as the pipeline (evaluate_alerts)
PSI_ATTENTION, PSI_THRESHOLD = 0.1, 0.2
KS_SHARE_WARN = 0.25
AUC_WARN_DROP, AUC_CRIT_DROP = 0.03, 0.05

# Baseline (held-out test split, threshold 0.5) from reports/baseline_metrics.json
BASELINE_AUC, BASELINE_RECALL, BASELINE_DEFAULT = 0.791, 0.382, 0.221

# Causal analysis (reports/causal_analysis.md): 30 seeds, period-4 intensity
CAUSAL = [
    ("control", 0.779, 0.0),
    ("inflation only", 0.777, -0.002),
    ("more late payments only", 0.772, -0.007),
    ("younger customers only", 0.778, -0.001),
    ("data drift only (all three)", 0.768, -0.011),
    ("concept drift only", 0.651, -0.128),
    ("everything (period 4)", 0.653, -0.125),
]
CAUSAL_PT = [
    "controle (nenhuma causa)", "só inflação", "só mais atrasos",
    "só clientes mais jovens", "só data drift (as 3 causas)",
    "só concept drift", "tudo (período 4)",
]

TXT = {
    "en": {
        "title": "Credit Scoring Observability",
        "sub": "Interactive replay of the monitoring results. Production data is simulated on the UCI "
               "Default of Credit Card Clients dataset (Taiwan, 2005).",
        "lang": "Language",
        "period": "Production period",
        "period_help": "Period 1 is a control (no drift). Intensity grows up to period 4.",
        "auc": "Model AUC", "recall": "Recall", "default": "Default rate", "psimax": "Max PSI",
        "vs_base": "vs baseline",
        "tab_drift": "Drift and alerts", "tab_perf": "Performance", "tab_cause": "Root cause",
        "tab_fair": "Fairness",
        "psi_title": "Top 10 features by PSI",
        "psi_note": "PSI < 0.1 stable, 0.1 to 0.2 attention, >= 0.2 drift.",
        "alerts": "Alerts that would fire",
        "none": "No alerts: everything within limits.",
        "crit": "critical", "warn": "warning",
        "r_psi_c": "{f} PSI={v:.3f}", "r_psi_w": "{f} PSI={v:.3f}",
        "r_ks": "KS flags {s:.0%} of the features",
        "r_auc": "AUC dropped {d:.3f} vs baseline",
        "auc_title": "AUC per period vs baseline",
        "baseline": "baseline", "model": "model",
        "perf_note": "The AUC alert fires in period 2, one batch before the first critical PSI alert "
                     "(period 3): performance monitoring catches the degradation first.",
        "cause_title": "AUC change vs control, one cause at a time",
        "cause_note": "Input shifts together cost about 0.011 of AUC; concept drift alone costs 0.128. "
                      "A high PSI is a reason to investigate, not proof of performance loss. "
                      "Proportions depend on the intensities chosen in the simulator.",
        "fair_note": "Group-level metrics on the held-out test set (threshold 0.5). "
                     "Sex is excluded from the model; a proxy test still predicts it from the "
                     "model features with AUC 0.649 (control 0.498), a moderate signal. "
                     "No confidence intervals; the 'other' marital group has only 61 customers.",
        "group": "Group", "attribute": "Attribute",
        "footer": "Educational project (FIAP Tech Challenge, Phase 4). Not legal or financial advice. "
                  "Source code and full report: github.com/AllySantos/tech_challenge/tree/main/phases/phase-4",
        "scenario": "Scenario", "delta": "AUC change",
    },
    "pt": {
        "title": "Credit Scoring Observability",
        "sub": "Replay interativo dos resultados de monitoramento. Os dados de produção são simulados sobre "
               "a base UCI Default of Credit Card Clients (Taiwan, 2005).",
        "lang": "Idioma",
        "period": "Período de produção",
        "period_help": "O período 1 é controle (sem drift). A intensidade cresce até o período 4.",
        "auc": "AUC do modelo", "recall": "Recall", "default": "Inadimplência", "psimax": "PSI máximo",
        "vs_base": "vs baseline",
        "tab_drift": "Drift e alertas", "tab_perf": "Desempenho", "tab_cause": "Causa",
        "tab_fair": "Justiça",
        "psi_title": "Top 10 variáveis por PSI",
        "psi_note": "PSI < 0,1 estável, 0,1 a 0,2 atenção, >= 0,2 drift.",
        "alerts": "Alertas que disparariam",
        "none": "Sem alertas: tudo dentro dos limites.",
        "crit": "crítico", "warn": "atenção",
        "r_psi_c": "{f} PSI={v:.3f}", "r_psi_w": "{f} PSI={v:.3f}",
        "r_ks": "KS acusa {s:.0%} das variáveis",
        "r_auc": "AUC caiu {d:.3f} vs baseline",
        "auc_title": "AUC por período vs baseline",
        "baseline": "baseline", "model": "modelo",
        "perf_note": "O alerta de AUC dispara no período 2, um lote antes do primeiro alerta crítico de PSI "
                     "(período 3): monitorar o desempenho pega a degradação primeiro.",
        "cause_title": "Variação do AUC vs controle, uma causa por vez",
        "cause_note": "Todas as mudanças de entrada juntas custam cerca de 0,011 de AUC; só o concept drift "
                      "custa 0,128. PSI alto é motivo para investigar, não prova de perda de desempenho. "
                      "As proporções dependem das intensidades escolhidas no simulador.",
        "fair_note": "Métricas por grupo no conjunto de teste (limiar 0,5). O sexo está fora do modelo; "
                     "mesmo assim, um teste de proxy o prevê a partir das variáveis do modelo com AUC 0,649 "
                     "(controle 0,498), um sinal moderado. Sem intervalos de confiança; o grupo "
                     "estado civil 'other' tem só 61 clientes.",
        "group": "Grupo", "attribute": "Atributo",
        "footer": "Projeto educacional (FIAP Tech Challenge, Fase 4). Não é aconselhamento jurídico ou "
                  "financeiro. Código e relatório completo: github.com/AllySantos/tech_challenge/tree/main/phases/phase-4",
        "scenario": "Cenário", "delta": "Variação do AUC",
    },
}

BAND = alt.Scale(domain=["stable", "attention", "drift"], range=["#4c9f70", "#e0a030", "#e0505a"])


@st.cache_data
def load():
    periods = json.loads((DATA / "drift_summary.json").read_text(encoding="utf-8"))
    fair = json.loads((DATA / "fairness_audit.json").read_text(encoding="utf-8"))
    return periods, fair


def alerts_for(p):
    out = []
    for f, v in p["psi"].items():
        if v >= PSI_THRESHOLD:
            out.append(("critical", "r_psi_c", dict(f=f, v=v)))
        elif v >= PSI_ATTENTION:
            out.append(("warning", "r_psi_w", dict(f=f, v=v)))
    share = len(p["drifted_ks"]) / len(p["psi"])
    if share >= KS_SHARE_WARN:
        out.append(("warning", "r_ks", dict(s=share)))
    drop = BASELINE_AUC - p["auc"]
    if drop >= AUC_CRIT_DROP:
        out.append(("critical", "r_auc", dict(d=drop)))
    elif drop >= AUC_WARN_DROP:
        out.append(("warning", "r_auc", dict(d=drop)))
    return out


def band(v):
    return "drift" if v >= PSI_THRESHOLD else "attention" if v >= PSI_ATTENTION else "stable"


st.set_page_config(page_title="Credit Scoring Observability", page_icon="📉", layout="wide")
periods, fair = load()

lang = st.sidebar.radio("Language / Idioma", ["en", "pt"], format_func=lambda x: {"en": "English", "pt": "Português"}[x])
t = TXT[lang]
k = st.sidebar.select_slider(t["period"], options=[1, 2, 3, 4], value=4, help=t["period_help"])
p = periods[k - 1]

st.title(t["title"])
st.caption(t["sub"])

psi_max = max(p["psi"].values())
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["auc"], f"{p['auc']:.3f}", f"{p['auc'] - BASELINE_AUC:+.3f} {t['vs_base']}", delta_color="normal")
c2.metric(t["recall"], f"{p['recall']:.3f}", f"{p['recall'] - BASELINE_RECALL:+.3f} {t['vs_base']}")
c3.metric(t["default"], f"{p['default_rate']:.1%}", f"{(p['default_rate'] - BASELINE_DEFAULT) * 100:+.1f} pp", delta_color="inverse")
c4.metric(t["psimax"], f"{psi_max:.3f}")

tab_d, tab_p, tab_c, tab_f = st.tabs([t["tab_drift"], t["tab_perf"], t["tab_cause"], t["tab_fair"]])

with tab_d:
    left, right = st.columns([3, 2])
    with left:
        st.subheader(t["psi_title"])
        top = (pd.Series(p["psi"]).sort_values(ascending=False).head(10).rename("psi").reset_index()
               .rename(columns={"index": "feature"}))
        top["band"] = top["psi"].map(band)
        bars = alt.Chart(top).mark_bar().encode(
            x=alt.X("psi:Q", title="PSI", scale=alt.Scale(domain=[0, max(0.6, top["psi"].max() * 1.1)])),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("band:N", scale=BAND, legend=None),
            tooltip=["feature", alt.Tooltip("psi:Q", format=".3f")],
        )
        rules = alt.Chart(pd.DataFrame({"x": [PSI_ATTENTION, PSI_THRESHOLD]})).mark_rule(
            strokeDash=[4, 4], color="#8a94a6").encode(x="x:Q")
        st.altair_chart(bars + rules, width="stretch")
        st.caption(t["psi_note"])
    with right:
        st.subheader(t["alerts"])
        al = alerts_for(p)
        if not al:
            st.success(t["none"])
        for sev, key, ctx in al:
            (st.error if sev == "critical" else st.warning)(f"**{t['crit' if sev == 'critical' else 'warn']}** · {t[key].format(**ctx)}")

with tab_p:
    st.subheader(t["auc_title"])
    df = pd.DataFrame({"period": [p_["period"] for p_ in periods], "AUC": [p_["auc"] for p_ in periods]})
    line = alt.Chart(df).mark_line(point=True, color="#76a7ff").encode(
        x=alt.X("period:O", title=None), y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0.6, 0.82])),
        tooltip=["period", alt.Tooltip("AUC:Q", format=".3f")])
    base = alt.Chart(pd.DataFrame({"y": [BASELINE_AUC]})).mark_rule(strokeDash=[5, 4], color="#e0a030").encode(y="y:Q")
    mark = alt.Chart(df[df["period"] == k]).mark_point(size=220, filled=True, color="#e0505a").encode(x="period:O", y="AUC:Q")
    st.altair_chart(line + base + mark, width="stretch")
    st.caption(f"{t['baseline']}: {BASELINE_AUC:.3f}. " + t["perf_note"])

with tab_c:
    st.subheader(t["cause_title"])
    names = [n for n, _, _ in CAUSAL] if lang == "en" else CAUSAL_PT
    cdf = pd.DataFrame({t["scenario"]: names, t["delta"]: [d for _, _, d in CAUSAL]})
    st.altair_chart(alt.Chart(cdf).mark_bar(color="#76a7ff").encode(
        x=alt.X(f"{t['delta']}:Q"), y=alt.Y(f"{t['scenario']}:N", sort=None, title=None),
        tooltip=[t["scenario"], alt.Tooltip(f"{t['delta']}:Q", format=".3f")]), width="stretch")
    st.info(t["cause_note"])

with tab_f:
    g = pd.DataFrame(fair["groups"]).rename(columns={
        "attribute": t["attribute"], "group": t["group"], "actual_default": "default rate",
        "flag_rate": "flag rate"})
    g[t["attribute"]] = g[t["attribute"]].str.replace("_group", "", regex=False)
    st.dataframe(g, hide_index=True, width="stretch")
    st.caption(t["fair_note"])

st.divider()
st.caption(t["footer"])
