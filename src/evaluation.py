"""Business-oriented evaluation helpers for fraud alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Assumed cost of one analyst reviewing one alert (USD). A missed fraud costs its amount.
REVIEW_COST = 5.0


def alert_stats(flagged: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray,
                review_cost: float = REVIEW_COST) -> dict:
    """Summarise a set of alerts: volume, precision, recall and money impact."""
    flagged = np.asarray(flagged, dtype=bool)
    is_fraud = np.asarray(is_fraud, dtype=bool)
    amount = np.asarray(amount, dtype=float)

    alerts = int(flagged.sum())
    caught = int((flagged & is_fraud).sum())
    total_fraud = int(is_fraud.sum())
    fraud_amt_caught = float(amount[flagged & is_fraud].sum())
    fraud_amt_total = float(amount[is_fraud].sum())
    return {
        "alerts": alerts,
        "fraud_caught": caught,
        "precision": caught / alerts if alerts else 0.0,
        "recall": caught / total_fraud if total_fraud else 0.0,
        "fraud_amt_caught": round(fraud_amt_caught, 2),
        "fraud_amt_recall": fraud_amt_caught / fraud_amt_total if fraud_amt_total else 0.0,
        "review_cost": round(alerts * review_cost, 2),
        "net_savings": round(fraud_amt_caught - alerts * review_cost, 2),
    }


def threshold_table(scores: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray,
                    thresholds: np.ndarray | None = None,
                    review_cost: float = REVIEW_COST) -> pd.DataFrame:
    """Alert statistics for a range of score thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        row = alert_stats(scores >= t, is_fraud, amount, review_cost)
        row["threshold"] = float(t)
        rows.append(row)
    cols = ["threshold"] + [c for c in rows[0] if c != "threshold"]
    return pd.DataFrame(rows)[cols]


def best_threshold(table: pd.DataFrame) -> float:
    """Threshold that maximises net savings (fraud $ stopped minus review cost)."""
    return float(table.loc[table["net_savings"].idxmax(), "threshold"])
