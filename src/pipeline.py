"""
End-to-end fraud detection pipeline.

    python -m src.pipeline --setup   # load CSVs, build SQL features & rules, then train + evaluate
    python -m src.pipeline           # re-run training + evaluation on existing features

Outputs (used by the README and the Streamlit dashboard):
    outputs/*.csv, outputs/metrics.json   -> results tables
    images/*.png                          -> charts
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.db import ROOT, query, run_sql_file
from src.evaluation import REVIEW_COST, alert_stats, best_threshold, threshold_table

OUT = ROOT / "outputs"
IMG = ROOT / "images"

NUMERIC = ["amt", "hour", "day_of_week", "customer_age", "city_pop", "distance_km",
           "card_txn_number", "secs_since_prev_txn", "txn_count_1h", "txn_count_24h",
           "amt_sum_24h", "card_avg_amt_prior", "amt_to_card_avg", "amt_zscore_in_category"]
BOOLEAN = ["is_night", "is_new_merchant"]
CATEGORICAL = ["category", "gender"]
FEATURES = NUMERIC + BOOLEAN + CATEGORICAL
RULES = {
    "r1_night_high_amount": "Night-time purchase ≥ $300",
    "r2_spike_vs_card_history": "≥ 5x the card's average spend",
    "r3_24h_spend_burst": "≥ $1,500 across 3+ transactions in 24h",
    "r4_category_outlier": "Amount ≥ 3 std devs above category norm",
}

BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    df = query("""
        SELECT f.*, r.r1_night_high_amount, r.r2_spike_vs_card_history,
               r.r3_24h_spend_burst, r.r4_category_outlier
        FROM fraud.features f
        JOIN fraud.rule_flags r USING (trans_num)
    """, parse_dates=["trans_ts"])
    for col in BOOLEAN + list(RULES):
        df[col] = df[col].map({"t": True, "f": False, True: True, False: False}).fillna(False).astype(bool)
    return df


def eda_tables(df: pd.DataFrame) -> dict:
    df = df.assign(fraud_amt=df["amt"] * df["is_fraud"])
    tables = {
        "fraud_by_hour": df.groupby("hour").agg(transactions=("is_fraud", "size"),
                                                fraud_cases=("is_fraud", "sum")),
        "fraud_by_category": df.groupby("category").agg(transactions=("is_fraud", "size"),
                                                        fraud_cases=("is_fraud", "sum"),
                                                        fraud_amt=("fraud_amt", "sum")),
    }
    bands = pd.cut(df["amt"], [0, 50, 100, 200, 300, 500, 1000, 1e9],
                   labels=["<$50", "$50-100", "$100-200", "$200-300", "$300-500", "$500-1k", "$1k+"])
    tables["fraud_by_amount_band"] = df.groupby(bands, observed=True).agg(
        transactions=("is_fraud", "size"), fraud_cases=("is_fraud", "sum"))
    for t in tables.values():
        t["fraud_rate_pct"] = (100 * t["fraud_cases"] / t["transactions"]).round(3)
    tables["fraud_by_category"] = tables["fraud_by_category"].sort_values("fraud_rate_pct", ascending=False)
    return {k: v.reset_index() for k, v in tables.items()}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_models() -> dict:
    lr = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC + BOOLEAN),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    hgb = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("bool", "passthrough", BOOLEAN),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("model", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            class_weight="balanced", random_state=42)),
    ])
    return {"Logistic regression (baseline)": lr, "Gradient boosting": hgb}


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X[BOOLEAN] = X[BOOLEAN].astype(float)
    return X


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_fraud_by_hour(t: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = [ORANGE if (h >= 22 or h < 4) else BLUE for h in t["hour"]]
    ax.bar(t["hour"], t["fraud_rate_pct"], color=colors, width=0.75)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Fraud rate by hour of day (10pm–4am highlighted)")
    ax.grid(axis="x", visible=False)
    save(fig, "01_fraud_rate_by_hour.png")


def chart_fraud_by_category(t: pd.DataFrame):
    d = t.sort_values("fraud_rate_pct")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(d["category"].str.replace("_", " "), d["fraud_rate_pct"], color=BLUE, height=0.7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Fraud rate by merchant category")
    ax.grid(axis="y", visible=False)
    save(fig, "02_fraud_rate_by_category.png")


def chart_pr_curves(curves: dict, rule_points: pd.DataFrame, chosen: dict):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    palette = {"Gradient boosting": BLUE, "Logistic regression (baseline)": GRAY}
    for name, (p, r, ap) in curves.items():
        ax.plot(r, p, color=palette[name], lw=2, label=f"{name} (PR-AUC {ap:.2f})")
    ax.scatter(rule_points["recall"], rule_points["precision"], s=60, color=ORANGE,
               zorder=3, label="Hand-written rules", edgecolor="white", linewidth=1.5)
    ax.scatter([chosen["recall"]], [chosen["precision"]], s=110, marker="*", color=INK, zorder=4,
               label=f"Chosen threshold ({chosen['threshold']:.2f})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Model vs rules on unseen test period")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    save(fig, "03_precision_recall.png")


def chart_savings(table: pd.DataFrame, best: float):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(table["threshold"], table["net_savings"] / 1e3, color=BLUE, lw=2, marker="o", markersize=4)
    row = table.loc[table["threshold"] == best].iloc[0]
    ax.annotate(f"Best: threshold {best:.2f}\n\\${row['net_savings'] / 1e3:,.0f}k net saved",
                (best, row["net_savings"] / 1e3), xytext=(40, -60), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("Alert threshold (model score)")
    ax.set_ylabel("Net savings (\\$ thousands)")
    ax.set_title(f"Fraud \\$ stopped minus review cost (\\${REVIEW_COST:.0f} per alert)")
    save(fig, "04_net_savings_by_threshold.png")


def chart_importance(imp: pd.DataFrame):
    d = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(d["feature"], d["importance"], color=BLUE, height=0.7)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("What the model relies on (permutation importance)")
    ax.grid(axis="y", visible=False)
    save(fig, "05_feature_importance.png")


# ----------------------------------------------------------------------------
def reasons(row) -> str:
    r = [label for col, label in RULES.items() if row[col]]
    if row["txn_count_1h"] >= 3:
        r.append(f"{int(row['txn_count_1h'])} transactions in the last hour")
    if row["distance_km"] >= 120:
        r.append(f"Merchant {row['distance_km']:.0f} km from home")
    return "; ".join(r) if r else "Model pattern (no single rule)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="load CSVs and build SQL features first")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(exist_ok=True)

    if args.setup:
        for f in ["01_create_tables.sql", "02_load.sql", "03_features.sql", "04_rules.sql"]:
            log(f"running sql/{f}")
            print(run_sql_file(f"sql/{f}"))

    log("loading features")
    df = load_features()
    train, test = df[df.dataset == "train"], df[df.dataset == "test"]
    y_train, y_test = train["is_fraud"].values, test["is_fraud"].values
    log(f"train {len(train):,} rows ({y_train.mean():.3%} fraud) | test {len(test):,} rows ({y_test.mean():.3%} fraud)")

    # --- EDA ---------------------------------------------------------------
    eda = eda_tables(df)
    for name, t in eda.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
    chart_fraud_by_hour(eda["fraud_by_hour"])
    chart_fraud_by_category(eda["fraud_by_category"])

    # --- Rules (evaluated on the test period) ------------------------------
    rule_rows = []
    for col, label in RULES.items():
        rule_rows.append({"rule": col, "description": label,
                          **alert_stats(test[col].values, y_test, test["amt"].values)})
    any_rule = test[list(RULES)].any(axis=1).values
    rule_rows.append({"rule": "any_rule", "description": "Any rule fires",
                      **alert_stats(any_rule, y_test, test["amt"].values)})
    rules_perf = pd.DataFrame(rule_rows)
    rules_perf.to_csv(OUT / "rule_performance.csv", index=False)
    log("rules evaluated")

    # --- Models ------------------------------------------------------------
    X_train, X_test = prepare_X(train), prepare_X(test)
    curves, model_metrics, fitted = {}, {}, {}
    for name, model in build_models().items():
        log(f"training {name}")
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        ap, auc = average_precision_score(y_test, scores), roc_auc_score(y_test, scores)
        p, r, _ = precision_recall_curve(y_test, scores)
        curves[name] = (p, r, ap)
        model_metrics[name] = {"pr_auc": round(ap, 4), "roc_auc": round(auc, 4)}
        fitted[name] = (model, scores)
        log(f"  PR-AUC {ap:.4f} | ROC-AUC {auc:.4f}")

    best_model, scores = fitted["Gradient boosting"]
    thresholds = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.99]]), 2)
    table = threshold_table(scores, y_test, test["amt"].values, thresholds)
    table.to_csv(OUT / "threshold_table.csv", index=False)
    best = best_threshold(table)
    chosen = table.loc[table["threshold"] == best].iloc[0].to_dict()
    for k in ("alerts", "fraud_caught"):
        chosen[k] = int(chosen[k])

    chart_pr_curves(curves, rules_perf[rules_perf.rule != "any_rule"], chosen)
    chart_savings(table, best)

    # PR curve points for the dashboard (downsampled)
    pr = pd.concat([pd.DataFrame({"model": n, "precision": p, "recall": r}).iloc[:: max(1, len(p) // 400)]
                    for n, (p, r, _) in curves.items()])
    pr.to_csv(OUT / "pr_curves.csv", index=False)

    # --- Feature importance (permutation on a stratified test sample) ---------
    log("permutation importance")
    sample = pd.concat([test[test.is_fraud == 1], test[test.is_fraud == 0].sample(min(60_000, int((test.is_fraud == 0).sum())), random_state=1)])
    pi = permutation_importance(best_model, prepare_X(sample), sample["is_fraud"].values,
                                scoring="average_precision", n_repeats=3, random_state=1, n_jobs=1)
    imp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean})
           .sort_values("importance", ascending=False))
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    chart_importance(imp)

    # --- Flagged transactions for the dashboard ------------------------------
    scored = test.assign(fraud_score=scores)
    top = scored[scored.fraud_score >= best].sort_values("fraud_score", ascending=False).head(3000).copy()
    top["reasons"] = top.apply(reasons, axis=1)
    top[["trans_num", "trans_ts", "category", "amt", "state", "hour", "distance_km",
         "txn_count_24h", "amt_to_card_avg", "fraud_score", "is_fraud", "reasons"]] \
        .round({"fraud_score": 4, "amt_to_card_avg": 2}) \
        .to_csv(OUT / "flagged_transactions.csv", index=False)

    # --- Headline metrics ------------------------------------------------------
    any_rule_stats = rules_perf[rules_perf.rule == "any_rule"].iloc[0].to_dict()
    metrics = {
        "data": {
            "transactions": int(len(df)), "cards": int(df.card_id.nunique()),
            "fraud_rate_pct": round(100 * df.is_fraud.mean(), 3),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_period": [str(train.trans_ts.min().date()), str(train.trans_ts.max().date())],
            "test_period": [str(test.trans_ts.min().date()), str(test.trans_ts.max().date())],
            "test_fraud_cases": int(y_test.sum()),
            "test_fraud_amt": round(float(test.loc[test.is_fraud == 1, "amt"].sum()), 2),
        },
        "models": model_metrics,
        "chosen_threshold": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in chosen.items()},
        "rules_any": {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in any_rule_stats.items() if k not in ("rule", "description")},
        "review_cost_per_alert": REVIEW_COST,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log("done")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
