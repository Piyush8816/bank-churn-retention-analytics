"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Streamlit Web Application

Core Modules:
  1. Engagement vs Churn Overview
  2. Product Utilization Impact Analysis
  3. High-Value Disengaged Customer Detector
  4. Retention Strength Scoring Panels

User Capabilities: Engagement filters, product count sliders, balance/salary thresholds
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

from feature_engineering import build_full_feature_set, compute_kpis

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = build_full_feature_set(os.path.join(BASE_DIR, "European_Bank.csv"))
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model", "churn_model.joblib")
    cols_path = os.path.join(BASE_DIR, "model", "feature_columns.joblib")
    results_path = os.path.join(BASE_DIR, "model", "results.json")
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    with open(results_path) as f:
        results = json.load(f)
    return model, feature_cols, results

df_full = load_data()
model, feature_cols, model_results = load_model()

# ----------------------------------------------------------------------------
# Sidebar — User Capabilities: filters
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Retention Analytics")
st.sidebar.markdown("### Filters")

geo_filter = st.sidebar.multiselect(
    "Geography", options=sorted(df_full["Geography"].unique()),
    default=sorted(df_full["Geography"].unique())
)

engagement_filter = st.sidebar.multiselect(
    "Engagement Segment",
    options=sorted(df_full["EngagementSegment"].unique()),
    default=sorted(df_full["EngagementSegment"].unique())
)

product_range = st.sidebar.slider(
    "Number of Products", min_value=int(df_full["NumOfProducts"].min()),
    max_value=int(df_full["NumOfProducts"].max()),
    value=(int(df_full["NumOfProducts"].min()), int(df_full["NumOfProducts"].max()))
)

balance_range = st.sidebar.slider(
    "Balance Range (€)",
    min_value=float(df_full["Balance"].min()), max_value=float(df_full["Balance"].max()),
    value=(float(df_full["Balance"].min()), float(df_full["Balance"].max()))
)

salary_range = st.sidebar.slider(
    "Estimated Salary Range (€)",
    min_value=float(df_full["EstimatedSalary"].min()), max_value=float(df_full["EstimatedSalary"].max()),
    value=(float(df_full["EstimatedSalary"].min()), float(df_full["EstimatedSalary"].max()))
)

active_filter = st.sidebar.radio("Activity Status", options=["All", "Active only", "Inactive only"])

# Apply filters
df = df_full[
    (df_full["Geography"].isin(geo_filter)) &
    (df_full["EngagementSegment"].isin(engagement_filter)) &
    (df_full["NumOfProducts"].between(product_range[0], product_range[1])) &
    (df_full["Balance"].between(balance_range[0], balance_range[1])) &
    (df_full["EstimatedSalary"].between(salary_range[0], salary_range[1]))
]
if active_filter == "Active only":
    df = df[df["IsActiveMember"] == 1]
elif active_filter == "Inactive only":
    df = df[df["IsActiveMember"] == 0]

st.sidebar.markdown(f"**{len(df):,}** customers match current filters (of {len(df_full):,} total)")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Customer Engagement & Product Utilization Analytics")
st.caption("Retention Strategy Dashboard — European Bank Customer Base")

if len(df) == 0:
    st.warning("No customers match the current filter selection. Please broaden your filters.")
    st.stop()

# Top-level KPI strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers (filtered)", f"{len(df):,}")
c2.metric("Churn Rate", f"{df['Exited'].mean()*100:.1f}%")
c3.metric("Active Members", f"{(df['IsActiveMember']==1).mean()*100:.1f}%")
c4.metric("Avg. Products / Customer", f"{df['NumOfProducts'].mean():.2f}")
c5.metric("Avg. Balance", f"€{df['Balance'].mean():,.0f}")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Engagement vs Churn",
    "🧩 Product Utilization",
    "💎 High-Value Disengaged Detector",
    "🔗 Retention Strength Scoring",
    "🔮 Churn Risk Predictor"
])

# ============================================================================
# MODULE 1: Engagement vs Churn Overview
# ============================================================================
with tab1:
    st.header("Engagement vs Churn Overview")
    st.markdown(
        "Compares churn outcomes across the four engagement profiles defined in the "
        "methodology: **Active Engaged**, **Active Low-Product**, **Inactive Disengaged**, "
        "and **Inactive High-Balance** customers."
    )

    col1, col2 = st.columns([1.3, 1])
    with col1:
        seg_churn = df.groupby("EngagementSegment")["Exited"].agg(["mean", "count"]).reset_index()
        seg_churn.columns = ["Segment", "ChurnRate", "Count"]
        seg_churn = seg_churn.sort_values("ChurnRate", ascending=False)
        fig = px.bar(seg_churn, x="Segment", y="ChurnRate", color="Segment",
                     text=seg_churn["ChurnRate"].apply(lambda x: f"{x*100:.1f}%"),
                     title="Churn Rate by Engagement Segment")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_dist = df["EngagementSegment"].value_counts().reset_index()
        seg_dist.columns = ["Segment", "Count"]
        fig2 = px.pie(seg_dist, names="Segment", values="Count", title="Customer Base Composition", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Active vs Inactive: The Core Engagement Signal")
    col3, col4 = st.columns(2)
    with col3:
        act = df.groupby("IsActiveMember")["Exited"].mean().rename({0: "Inactive", 1: "Active"})
        fig3 = px.bar(x=act.index, y=act.values, labels={"x": "Status", "y": "Churn Rate"},
                      color=act.index, text=[f"{v*100:.1f}%" for v in act.values],
                      title="Churn Rate: Active vs Inactive Members")
        fig3.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        active_rate = df.loc[df.IsActiveMember == 1, "Exited"].mean()
        inactive_rate = df.loc[df.IsActiveMember == 0, "Exited"].mean()
        ratio = inactive_rate / active_rate if active_rate > 0 else np.nan
        st.metric("Engagement Retention Ratio", f"{ratio:.2f}x",
                   help="Inactive churn rate ÷ Active churn rate. A value of 2.0 means inactive customers churn twice as often.")
        st.info(
            f"Inactive customers churn at **{inactive_rate*100:.1f}%** vs **{active_rate*100:.1f}%** "
            f"for active customers — roughly **{ratio:.1f}x** higher. Engagement status is one of the "
            f"strongest retention signals in this data, often outweighing balance or salary."
        )

    st.subheader("Geography & Gender Breakdown")
    col5, col6 = st.columns(2)
    with col5:
        geo_churn = df.groupby("Geography")["Exited"].mean().reset_index()
        fig4 = px.bar(geo_churn, x="Geography", y="Exited", title="Churn Rate by Geography",
                      text=geo_churn["Exited"].apply(lambda x: f"{x*100:.1f}%"))
        fig4.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)
    with col6:
        gen_churn = df.groupby("Gender")["Exited"].mean().reset_index()
        fig5 = px.bar(gen_churn, x="Gender", y="Exited", title="Churn Rate by Gender",
                      text=gen_churn["Exited"].apply(lambda x: f"{x*100:.1f}%"))
        fig5.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)

# ============================================================================
# MODULE 2: Product Utilization Impact Analysis
# ============================================================================
with tab2:
    st.header("Product Utilization Impact Analysis")
    st.markdown(
        "Examines how the **number and depth of products held** relates to churn, "
        "and compares single-product vs multi-product retention."
    )

    col1, col2 = st.columns(2)
    with col1:
        prod_churn = df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"]).reset_index()
        prod_churn.columns = ["NumOfProducts", "ChurnRate", "Count"]
        fig = px.bar(prod_churn, x="NumOfProducts", y="ChurnRate",
                     text=prod_churn["ChurnRate"].apply(lambda x: f"{x*100:.1f}%"),
                     title="Churn Rate by Number of Products")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Sample sizes — {dict(zip(prod_churn.NumOfProducts, prod_churn.Count))}")

    with col2:
        single_multi = df.groupby("IsMultiProduct")["Exited"].mean().rename({0: "Single Product", 1: "Multi-Product (2+)"})
        fig2 = px.bar(x=single_multi.index, y=single_multi.values,
                      labels={"x": "", "y": "Churn Rate"},
                      text=[f"{v*100:.1f}%" for v in single_multi.values],
                      title="Single-Product vs Multi-Product Retention")
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "⚠️ **Non-obvious finding:** churn rises sharply for customers holding **3 or 4 products** "
        "(often 80–100% churn in this segment), despite being a small group. This does *not* mean "
        "more products cause churn — it more likely reflects a small population of over-sold, "
        "dissatisfied, or already-exiting customers who accumulated products shortly before leaving. "
        "This deserves qualitative follow-up (e.g. complaint records) before acting on it."
    )

    st.subheader("Product Depth vs Engagement")
    depth_eng = df.groupby(["ProductDepthTier", "IsActiveMember"])["Exited"].mean().reset_index()
    depth_eng["IsActiveMember"] = depth_eng["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    fig3 = px.bar(depth_eng, x="ProductDepthTier", y="Exited", color="IsActiveMember", barmode="group",
                  title="Churn Rate: Product Depth × Activity Status",
                  labels={"Exited": "Churn Rate"})
    fig3.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# MODULE 3: High-Value Disengaged Customer Detector
# ============================================================================
with tab3:
    st.header("High-Value Disengaged Customer Detector")
    st.markdown(
        "Surfaces customers who look financially strong (high balance / salary) but show "
        "**low engagement** — the exact profile the project brief identifies as a silent churn risk."
    )

    colA, colB, colC = st.columns(3)
    balance_pctile = colA.slider("Balance percentile threshold", 50, 99, 75,
                                  help="Customers above this balance percentile are considered 'high-value'")
    require_inactive = colB.checkbox("Require: Inactive member", value=True)
    require_low_product = colC.checkbox("Require: ≤1 product", value=False)

    threshold_balance = df_full["Balance"].quantile(balance_pctile / 100)
    at_risk = df[df["Balance"] >= threshold_balance]
    if require_inactive:
        at_risk = at_risk[at_risk["IsActiveMember"] == 0]
    if require_low_product:
        at_risk = at_risk[at_risk["NumOfProducts"] <= 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("High-Value Disengaged Customers", f"{len(at_risk):,}")
    col2.metric("Their Churn Rate", f"{at_risk['Exited'].mean()*100:.1f}%" if len(at_risk) else "N/A")
    col3.metric("Total Balance at Risk", f"€{at_risk['Balance'].sum():,.0f}")

    if len(at_risk) > 0:
        st.dataframe(
            at_risk[["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts",
                     "IsActiveMember", "HasCrCard", "EstimatedSalary", "RelationshipStrengthIndex", "Exited"]]
            .sort_values("Balance", ascending=False)
            .rename(columns={"Exited": "AlreadyChurned"}),
            use_container_width=True, height=400
        )
        st.download_button(
            "⬇️ Download this customer list (CSV)",
            data=at_risk.to_csv(index=False).encode("utf-8"),
            file_name="high_value_disengaged_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("No customers match these thresholds — try lowering the balance percentile.")

# ============================================================================
# MODULE 4: Retention Strength Scoring Panels
# ============================================================================
with tab4:
    st.header("Retention Strength Scoring")
    st.markdown(
        "The **Relationship Strength Index** (0–4) combines activity status, credit card "
        "ownership, multi-product holding, and above-median tenure into one score. "
        "Customers scoring 3+ are classified as **'sticky'**."
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        rsi = df.groupby("RelationshipStrengthIndex")["Exited"].agg(["mean", "count"]).reset_index()
        rsi.columns = ["RSI", "ChurnRate", "Count"]
        fig = px.bar(rsi, x="RSI", y="ChurnRate", text=rsi["ChurnRate"].apply(lambda x: f"{x*100:.1f}%"),
                     title="Churn Rate by Relationship Strength Index")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sticky_rate = df.loc[df.IsStickyCustomer == 1, "Exited"].mean()
        non_sticky_rate = df.loc[df.IsStickyCustomer == 0, "Exited"].mean()
        st.metric("Sticky Customers (RSI ≥ 3)", f"{(df.IsStickyCustomer==1).sum():,} ({(df.IsStickyCustomer==1).mean()*100:.1f}%)")
        st.metric("Sticky Customer Churn Rate", f"{sticky_rate*100:.1f}%")
        st.metric("Non-Sticky Customer Churn Rate", f"{non_sticky_rate*100:.1f}%")

    st.subheader("Credit Card Stickiness")
    cc = df.groupby("HasCrCard")["Exited"].mean().rename({0: "No Card", 1: "Has Card"})
    fig2 = px.bar(x=cc.index, y=cc.values, labels={"x": "", "y": "Churn Rate"},
                  text=[f"{v*100:.1f}%" for v in cc.values], title="Churn Rate: Credit Card Ownership")
    fig2.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "In this dataset, credit card ownership shows little independent effect on churn once "
        "activity and product depth are accounted for — engagement and product depth are the "
        "stronger retention levers."
    )

    st.subheader("At-Risk Premium Customers")
    st.markdown("High balance + inactive + ≤1 product — the profile most likely to churn silently.")
    premium_risk = df[df["AtRiskPremium"] == 1]
    col3, col4 = st.columns(2)
    col3.metric("Count", f"{len(premium_risk):,}")
    col4.metric("Churn Rate", f"{premium_risk['Exited'].mean()*100:.1f}%" if len(premium_risk) else "N/A")

# ============================================================================
# MODULE 5 (bonus): Churn Risk Predictor — uses the trained ML model
# ============================================================================
with tab5:
    st.header("Individual Customer Churn Risk Predictor")
    st.caption(
        f"Powered by the trained model ({model_results['best_model']}) — "
        f"test accuracy {model_results['all_results'][model_results['best_model']]['accuracy']*100:.1f}%, "
        f"ROC-AUC {model_results['all_results'][model_results['best_model']]['roc_auc']:.3f}"
    )

    st.markdown("Enter a customer's profile to estimate churn probability:")
    colA, colB, colC = st.columns(3)
    with colA:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", sorted(df_full["Geography"].unique()))
        gender = st.selectbox("Gender", sorted(df_full["Gender"].unique()))
    with colB:
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0)
    with colC:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"]) == "Yes"
        is_active = st.selectbox("Active Member", ["Yes", "No"]) == "Yes"
        salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 100000.0, step=1000.0)

    if st.button("Predict Churn Risk", type="primary"):
        row = pd.DataFrame([{
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": int(has_card), "IsActiveMember": int(is_active), "EstimatedSalary": salary,
            "CustomerId": 0, "Surname": "temp", "Year": 2025, "Exited": 0
        }])
        from feature_engineering import engagement_classification, product_utilization_features, financial_commitment_features, retention_strength_features
        row = engagement_classification(row)
        row = product_utilization_features(row)
        row = financial_commitment_features(row)
        row = retention_strength_features(row)
        row = row.drop(columns=["Year", "CustomerId", "Surname", "Exited"])
        row = pd.get_dummies(row, columns=["Geography", "Gender", "ProductDepthTier", "EngagementSegment"], drop_first=True)
        row.columns = [str(c).replace('(', '').replace(')', '').replace(' ', '_').replace('-', '_').replace(',', '') for c in row.columns]
        row = row.reindex(columns=feature_cols, fill_value=0)

        proba = model.predict_proba(row)[0, 1]
        pred = "HIGH RISK — likely to churn" if proba >= 0.5 else "LOW RISK — likely to stay"

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{proba*100:.1f}%")
        with col2:
            if proba >= 0.5:
                st.error(pred)
            else:
                st.success(pred)

st.divider()
st.caption(
    "Customer Engagement & Product Utilization Analytics for Retention Strategy | "
    "Data source: European Bank customer dataset | Built with Streamlit"
)
