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
