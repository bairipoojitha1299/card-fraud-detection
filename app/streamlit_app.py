"""
Fraud monitoring dashboard.

    streamlit run app/streamlit_app.py

Reads only the small result files in outputs/, so it runs anywhere
(including Streamlit Community Cloud) without the database or raw data.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"

st.set_page_config(page_title="Card Fraud Monitor", page_icon="💳", layout="wide")


@st.cache_data
def load():
    metrics = json.loads((OUT / "metrics.json").read_text())
    tables = {name: pd.read_csv(OUT / f"{name}.csv") for name in [
        "fraud_by_hour", "fraud_by_category", "fraud_by_amount_band", "rule_performance",
        "threshold_table", "pr_curves", "feature_importance", "flagged_transactions"]}
    return metrics, tables


metrics, t = load()
data = metrics["data"]

st.title("💳 Credit Card Fraud Monitor")
st.caption(
    f"{data['transactions']:,} transactions from {data['cards']:,} cards. "
    f"The model was trained on {data['train_period'][0]} → {data['train_period'][1]} "
    f"and evaluated on the unseen period {data['test_period'][0]} → {data['test_period'][1]}."
)

tab_overview, tab_model, tab_alerts = st.tabs(["📊 Fraud patterns", "🎯 Model & threshold", "🚨 Alert queue"])

# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{data['transactions'] / 1e6:.2f}M")
    c2.metric("Fraud rate", f"{data['fraud_rate_pct']:.2f}%")
    c3.metric("Test-period fraud cases", f"{data['test_fraud_cases']:,}")
    c4.metric("Test-period fraud losses", f"${data['test_fraud_amt'] / 1e6:.2f}M")

    left, right = st.columns(2)
    hour = t["fraud_by_hour"].assign(period=lambda d: d["hour"].map(
        lambda h: "Night (10pm–4am)" if h >= 22 or h < 4 else "Day"))
    left.subheader("Fraud rate by hour")
    left.altair_chart(
        alt.Chart(hour).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("hour:O", title="Hour of day"),
            y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
            color=alt.Color("period:N", scale=alt.Scale(domain=["Day", "Night (10pm–4am)"],
                                                        range=[BLUE, ORANGE]), title=None),
            tooltip=["hour", "transactions", "fraud_cases", "fraud_rate_pct"],
        ).properties(height=320),
        width="stretch",
    )

    right.subheader("Fraud rate by category")
    right.altair_chart(
        alt.Chart(t["fraud_by_category"]).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            y=alt.Y("category:N", sort="-x", title=None),
            x=alt.X("fraud_rate_pct:Q", title="Fraud rate (%)"),
            tooltip=["category", "transactions", "fraud_cases", "fraud_rate_pct",
                     alt.Tooltip("fraud_amt:Q", format="$,.0f", title="fraud $")],
        ).properties(height=320),
        width="stretch",
    )

    st.subheader("Fraud rate by transaction amount")
    st.altair_chart(
        alt.Chart(t["fraud_by_amount_band"]).mark_bar(color=BLUE, cornerRadiusTopLeft=3,
                                                     cornerRadiusTopRight=3).encode(
            x=alt.X("amt:N", sort=None, title="Amount band"),
            y=alt.Y("fraud_rate_pct:Q", title="Fraud rate (%)"),
            tooltip=["amt", "transactions", "fraud_cases", "fraud_rate_pct"],
        ).properties(height=260),
        width="stretch",
    )

# ---------------------------------------------------------------------------
with tab_model:
    m = metrics["models"]
    c1, c2 = st.columns(2)
    c1.metric("Gradient boosting PR-AUC", f"{m['Gradient boosting']['pr_auc']:.3f}",
              f"{m['Gradient boosting']['pr_auc'] - m['Logistic regression (baseline)']['pr_auc']:+.3f} vs baseline")
    c2.metric("Gradient boosting ROC-AUC", f"{m['Gradient boosting']['roc_auc']:.3f}")

    st.subheader("Choose an alert threshold")
    table = t["threshold_table"]
    best = metrics["chosen_threshold"]["threshold"]
    options = table["threshold"].round(2).tolist()
    threshold = st.select_slider("Alert when the model's fraud score is at least",
                                 options=options, value=round(best, 2))
    row = table.loc[table["threshold"].round(2) == threshold].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Alerts to review", f"{int(row['alerts']):,}")
    c2.metric("Precision", f"{row['precision']:.1%}")
    c3.metric("Fraud cases caught", f"{row['recall']:.1%}")
    c4.metric("Fraud $ caught", f"{row['fraud_amt_recall']:.1%}")
    c5.metric("Net savings", f"${row['net_savings'] / 1e3:,.0f}k")
    st.caption(f"Net savings = fraud dollars stopped − ${metrics['review_cost_per_alert']:.0f} "
               f"review cost per alert. Best threshold on the test period: **{best:.2f}**.")

    left, right = st.columns(2)
    left.subheader("Precision–recall: model vs rules")
    rules = t["rule_performance"].query("rule != 'any_rule'")
    pr_lines = alt.Chart(t["pr_curves"]).mark_line(strokeWidth=2).encode(
        x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=alt.Scale(
            domain=["Gradient boosting", "Logistic regression (baseline)"], range=[BLUE, GRAY]), title=None),
    )
    rule_pts = alt.Chart(rules).mark_point(filled=True, size=90, color=ORANGE).encode(
        x="recall:Q", y="precision:Q", tooltip=["description", "alerts", "precision", "recall"])
    left.altair_chart((pr_lines + rule_pts).properties(height=340), width="stretch")
    left.caption("Orange dots are the four hand-written rules.")

    right.subheader("What drives the model")
    right.altair_chart(
        alt.Chart(t["feature_importance"].head(12)).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("importance:Q", title="Drop in PR-AUC when shuffled"),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
        ).properties(height=340),
        width="stretch",
    )

    st.subheader("Rule performance (test period)")
    st.dataframe(
        t["rule_performance"][["description", "alerts", "fraud_caught", "precision", "recall",
                               "fraud_amt_caught", "net_savings"]],
        hide_index=True, width="stretch",
        column_config={
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "recall": st.column_config.NumberColumn(format="%.3f"),
            "fraud_amt_caught": st.column_config.NumberColumn("fraud $ caught", format="$%.0f"),
            "net_savings": st.column_config.NumberColumn("net savings", format="$%.0f"),
        },
    )

# ---------------------------------------------------------------------------
with tab_alerts:
    st.subheader("Highest-risk transactions in the test period")
    flagged = t["flagged_transactions"]
    c1, c2 = st.columns([1, 2])
    min_score = c1.slider("Minimum fraud score", 0.0, 1.0, float(round(best, 2)), 0.01)
    categories = c2.multiselect("Categories", sorted(flagged["category"].unique()))
    view = flagged[flagged["fraud_score"] >= min_score]
    if categories:
        view = view[view["category"].isin(categories)]
    st.caption(f"{len(view):,} alerts shown · {view['is_fraud'].mean():.1%} confirmed fraud"
               if len(view) else "No alerts match the filters.")
    st.dataframe(
        view, hide_index=True, width="stretch", height=520,
        column_config={
            "fraud_score": st.column_config.ProgressColumn("fraud score", min_value=0.0, max_value=1.0,
                                                           format="%.3f"),
            "amt": st.column_config.NumberColumn("amount", format="$%.2f"),
            "is_fraud": st.column_config.CheckboxColumn("confirmed fraud"),
        },
    )

st.divider()
st.caption("Data: Credit Card Transactions Fraud Detection Dataset (simulated with Sparkov) on Kaggle.")
