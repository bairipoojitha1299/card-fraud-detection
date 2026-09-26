"""Unit tests for the alert / cost evaluation helpers."""

import numpy as np
import pytest

from src.evaluation import alert_stats, best_threshold, threshold_table

IS_FRAUD = np.array([1, 1, 0, 0, 0, 1])
AMOUNT = np.array([500.0, 200.0, 20.0, 30.0, 40.0, 100.0])
SCORES = np.array([0.95, 0.60, 0.70, 0.10, 0.05, 0.20])


def test_alert_stats_counts_and_money():
    s = alert_stats(SCORES >= 0.5, IS_FRAUD, AMOUNT, review_cost=5)
    assert s["alerts"] == 3                       # 0.95, 0.60, 0.70
    assert s["fraud_caught"] == 2
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["fraud_amt_caught"] == 700.0
    assert s["fraud_amt_recall"] == pytest.approx(700 / 800)
    assert s["net_savings"] == 700 - 3 * 5


def test_no_alerts_does_not_divide_by_zero():
    s = alert_stats(np.zeros(6, dtype=bool), IS_FRAUD, AMOUNT)
    assert s["alerts"] == 0 and s["precision"] == 0.0 and s["recall"] == 0.0


def test_threshold_table_is_monotonic_in_alerts():
    table = threshold_table(SCORES, IS_FRAUD, AMOUNT, thresholds=np.array([0.1, 0.5, 0.9]))
    assert list(table["threshold"]) == [0.1, 0.5, 0.9]
    assert table["alerts"].is_monotonic_decreasing


def test_best_threshold_maximises_net_savings():
    table = threshold_table(SCORES, IS_FRAUD, AMOUNT, thresholds=np.array([0.1, 0.5, 0.9]),
                            review_cost=5)
    # 0.1 catches all $800 of fraud with 4 alerts -> best net savings
    assert best_threshold(table) == 0.1
