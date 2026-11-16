"""Business and evaluation maths checked against hand-computed examples."""

import numpy as np
import pytest

from app.core import metrics

# 10 customers, already in descending score order; 4 churners.
SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
LABELS = [1, 1, 0, 1, 0, 0, 0, 0, 0, 1]


def test_captured_and_lift_hand_computed():
    # Top 20% = first 2 customers -> 2 of 4 churners = 50%, lift 0.5 / 0.2 = 2.5
    assert metrics.captured_at(SCORES, LABELS, 0.2) == pytest.approx(0.5)
    assert metrics.lift_at(SCORES, LABELS, 0.2) == pytest.approx(2.5)
    # Top 40% = 4 customers -> 3 of 4 = 75%, lift 1.875
    assert metrics.captured_at(SCORES, LABELS, 0.4) == pytest.approx(0.75)
    assert metrics.lift_at(SCORES, LABELS, 0.4) == pytest.approx(1.875)
    # Everyone -> 100%, lift 1
    assert metrics.lift_at(SCORES, LABELS, 1.0) == pytest.approx(1.0)


def test_gains_table_hand_computed():
    g = metrics.gains_table(SCORES, LABELS, n_bins=5)
    assert g["customers"].tolist() == [2, 2, 2, 2, 2]
    assert g["outcomes"].tolist() == [2, 1, 0, 0, 1]
    # base rate 0.4: bin rates 1.0, 0.5, 0, 0, 0.5 -> lifts 2.5, 1.25, 0, 0, 1.25
    assert g["lift"].tolist() == pytest.approx([2.5, 1.25, 0.0, 0.0, 1.25])
    assert g["cum_captured"].tolist() == pytest.approx([0.5, 0.75, 0.75, 0.75, 1.0])


def test_gains_curve_shuffled_input_is_sorted_by_score():
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(SCORES))
    x, y = metrics.gains_curve(np.array(SCORES)[idx], np.array(LABELS)[idx])
    assert x[0] == 0 and y[0] == 0 and y[-1] == pytest.approx(1.0)
    assert y[2] == pytest.approx(0.5)  # after 2 of 10 customers


def test_retention_plan_hand_computed():
    p, value = [0.5, 0.2], [1000.0, 5000.0]
    plan, order = metrics.retention_plan(p, value, offer_cost=100, save_rate=0.5, rank_by="prob")
    assert order.tolist() == [0, 1]
    # k=1: revenue 0.5*1000*0.5 = 250, cost 100, net 150
    # k=2: revenue 250 + 0.2*5000*0.5 = 750, cost 200, net 550
    assert plan["revenue_saved"].tolist() == pytest.approx([250, 750])
    assert plan["net"].tolist() == pytest.approx([150, 550])
    assert plan["saved"].tolist() == pytest.approx([0.25, 0.35])
    assert metrics.best_k(plan, capacity=1) == 1
    assert metrics.best_k(plan, capacity=2) == 2

    by_value, order_v = metrics.retention_plan(p, value, offer_cost=100, save_rate=0.5, rank_by="value")
    assert order_v.tolist() == [1, 0]  # 0.2*5000 = 1000 beats 0.5*1000 = 500
    assert by_value["net"].iloc[0] == pytest.approx(0.2 * 5000 * 0.5 - 100)


def test_best_k_is_zero_when_nothing_pays():
    plan, _ = metrics.retention_plan([0.1, 0.1], [0, 0], offer_cost=10, save_rate=0.5)
    assert metrics.best_k(plan, capacity=2) == 0


def test_threshold_sweep_and_confusion_hand_computed():
    # threshold 0.55 flags 0.9, 0.8, 0.7, 0.6 -> labels 1, 1, 0, 1
    cm = metrics.confusion_at(LABELS, SCORES, 0.55)
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (3, 1, 1, 5)
    assert cm["precision"] == pytest.approx(0.75) and cm["recall"] == pytest.approx(0.75)
    sweep = metrics.threshold_sweep(LABELS, SCORES, [0.0, 0.55, 1.0])
    assert sweep[["tp", "fp", "fn", "tn"]].values.tolist() == [[4, 6, 0, 0], [3, 1, 1, 5], [0, 0, 4, 6]]


def test_threshold_cost_hand_computed():
    # cost = fp * 10 + fn * 100
    # t=0.00: fp 6, fn 0 -> 60 ; t=0.55: fp 1, fn 1 -> 110 ; t=0.75: fp 0, fn 2 -> 200 ; t=1: fn 4 -> 400
    table = metrics.threshold_cost(LABELS, SCORES, cost_fp=10, cost_fn=100, grid=[0.0, 0.55, 0.75, 1.0])
    assert table["cost"].tolist() == pytest.approx([60, 110, 200, 400])
    assert metrics.best_cost_threshold(table) == 0.0
    # Make false alarms expensive: cost = fp * 100 + fn * 10 -> 600, 110, 20, 40 -> best t = 0.75
    table = metrics.threshold_cost(LABELS, SCORES, cost_fp=100, cost_fn=10, grid=[0.0, 0.55, 0.75, 1.0])
    assert table["cost"].tolist() == pytest.approx([600, 110, 20, 40])
    assert metrics.best_cost_threshold(table) == 0.75


def test_calibration_and_brier():
    cal = metrics.calibration_table([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], n_bins=2)
    assert cal["mean_pred"].tolist() == pytest.approx([0.15, 0.8])
    assert cal["observed"].tolist() == pytest.approx([0.0, 1.0])
    # Brier: (0.01 + 0.04 + 0.09 + 0.01) / 4 = 0.0375
    assert metrics.brier([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]) == pytest.approx(0.0375)


def test_group_metrics_flags_weak_groups():
    # Group A is perfectly ranked (AUC 1), group B is inverted (AUC 0): B must be flagged.
    y = [0, 0, 1, 1] * 10 + [0, 0, 1, 1] * 10
    p = [0.1, 0.2, 0.8, 0.9] * 10 + [0.8, 0.9, 0.1, 0.2] * 10
    g = ["A"] * 40 + ["B"] * 40
    out = metrics.group_metrics(y, p, g, threshold=0.5, min_size=10).set_index("group")
    assert out.loc["A", "auc"] == pytest.approx(1.0) and not out.loc["A", "flagged"]
    assert out.loc["B", "auc"] == pytest.approx(0.0) and out.loc["B", "flagged"]
    assert out.loc["A", "recall"] == pytest.approx(1.0) and out.loc["B", "recall"] == pytest.approx(0.0)
    small = metrics.group_metrics([0, 1], [0.2, 0.8], ["C", "C"], 0.5, min_size=30)
    assert small.loc[0, "too_small"] and not small.loc[0, "flagged"]


def test_sample_size_hand_computed():
    # p1 = 0.20, p2 = 0.15, alpha 0.05 two-sided (z = 1.959964), power 0.80 (z = 0.841621)
    # pbar = 0.175 -> sqrt(2 * 0.175 * 0.825) = 0.537355 ; sqrt(0.16 + 0.1275) = 0.536190
    # n = (1.959964 * 0.537355 + 0.841621 * 0.536190)^2 / 0.05^2 = 2.263424 / 0.0025 = 905.37 -> 906
    assert metrics.sample_size_two_proportions(0.20, 0.15, 0.05, 0.80) == 906
    # One-sided needs fewer customers
    assert metrics.sample_size_two_proportions(0.20, 0.15, 0.05, 0.80, two_sided=False) < 906
    with pytest.raises(ValueError):
        metrics.sample_size_two_proportions(0.2, 0.2)


def test_two_proportion_test_hand_computed():
    # control 200/1000 = 0.20, treatment 150/1000 = 0.15 -> uplift 0.05
    # pooled p = 0.175, se = sqrt(0.175 * 0.825 * 0.002) = 0.016993 -> z = 2.9424, p = 0.00326
    # unpooled se = sqrt(0.00016 + 0.0001275) = 0.016956 -> CI 0.05 +/- 1.96 * 0.016956 = [0.01677, 0.08323]
    r = metrics.two_proportion_test(150, 1000, 200, 1000)
    assert r["uplift"] == pytest.approx(0.05)
    assert r["z"] == pytest.approx(2.9424, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.00326, abs=5e-5)
    assert r["ci_low"] == pytest.approx(0.01677, abs=1e-4) and r["ci_high"] == pytest.approx(0.08323, abs=1e-4)
    assert r["significant"] and r["relative_reduction"] == pytest.approx(0.25)


def test_power_inverts_sample_size():
    n = metrics.sample_size_two_proportions(0.20, 0.15, 0.05, 0.80)
    assert metrics.power_two_proportions(0.20, 0.15, n) == pytest.approx(0.80, abs=0.002)
    assert metrics.power_two_proportions(0.20, 0.15, n // 4) < 0.5
