import pytest

from credit_scoring_observability.pipeline import (
    AUC_CRIT_DROP,
    AUC_WARN_DROP,
    KS_SHARE_WARN,
    PSI_ATTENTION,
    PSI_THRESHOLD,
    evaluate_alerts,
)

BASE_AUC = 0.79


def severities(alerts):
    return [(sev, rule) for sev, rule, _ in alerts]


def test_no_alerts_when_everything_is_stable():
    alerts = evaluate_alerts({"age": 0.01}, 0.0, BASE_AUC, BASE_AUC)
    assert alerts == []


def test_psi_above_threshold_is_critical():
    alerts = evaluate_alerts({"age": PSI_THRESHOLD + 0.05}, 0.0, BASE_AUC, BASE_AUC)
    assert ("critical", "psi_drift") in severities(alerts)


def test_psi_between_attention_and_threshold_is_warning():
    mid = (PSI_ATTENTION + PSI_THRESHOLD) / 2
    alerts = evaluate_alerts({"age": mid}, 0.0, BASE_AUC, BASE_AUC)
    assert severities(alerts) == [("warning", "psi_attention")]


def test_psi_exactly_at_threshold_is_critical():
    alerts = evaluate_alerts({"age": PSI_THRESHOLD}, 0.0, BASE_AUC, BASE_AUC)
    assert ("critical", "psi_drift") in severities(alerts)


def test_one_alert_per_drifting_feature():
    psi = {"age": PSI_THRESHOLD + 0.1, "bill_amt1": PSI_THRESHOLD + 0.1, "pay_1": 0.0}
    alerts = evaluate_alerts(psi, 0.0, BASE_AUC, BASE_AUC)
    assert len([a for a in alerts if a[1] == "psi_drift"]) == 2


def test_ks_share_triggers_warning():
    alerts = evaluate_alerts({}, KS_SHARE_WARN, BASE_AUC, BASE_AUC)
    assert severities(alerts) == [("warning", "ks_drift")]


def test_auc_drop_critical():
    alerts = evaluate_alerts({}, 0.0, BASE_AUC - AUC_CRIT_DROP - 0.001, BASE_AUC)
    assert severities(alerts) == [("critical", "auc_drop")]


def test_auc_drop_warning_not_critical():
    auc = BASE_AUC - (AUC_WARN_DROP + AUC_CRIT_DROP) / 2
    alerts = evaluate_alerts({}, 0.0, auc, BASE_AUC)
    assert severities(alerts) == [("warning", "auc_drop")]


def test_auc_gain_never_alerts():
    alerts = evaluate_alerts({}, 0.0, BASE_AUC + 0.05, BASE_AUC)
    assert alerts == []


@pytest.mark.parametrize("psi_value", [0.0, 0.05])
def test_small_psi_is_silent(psi_value):
    assert evaluate_alerts({"age": psi_value}, 0.0, BASE_AUC, BASE_AUC) == []
