"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
Feature Engineering Module
Implements the Analytical Methodology defined in the project brief:
1. Data Ingestion & Validation
2. Engagement Classification
3. Product Utilization Analysis
4. Financial Commitment vs Engagement Analysis
5. Retention Strength Assessment
"""

import pandas as pd
import numpy as np


def load_and_validate(path):
    """Step 1: Data Ingestion & Validation"""
    df = pd.read_csv(path)

    # Validate binary variable consistency
    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        assert set(df[col].unique()).issubset({0, 1}), f"{col} has non-binary values"

    # Confirm no missing values / duplicate customers
    assert df["CustomerId"].is_unique, "Duplicate CustomerId found"
    assert df.isnull().sum().sum() == 0, "Missing values found"

    # Basic range sanity checks
    assert df["Age"].between(18, 100).all()
    assert df["NumOfProducts"].between(1, 4).all()
    assert df["Tenure"].between(0, 10).all()

    return df


def engagement_classification(df):
    """Step 2: Engagement Classification
    Creates 4 engagement profiles:
      - Active engaged customers
      - Inactive disengaged customers
      - Active but low-product customers
      - Inactive high-balance customers
    """
    df = df.copy()
    median_balance = df["Balance"].median()

    def classify(row):
        active = row["IsActiveMember"] == 1
        low_product = row["NumOfProducts"] <= 1
        high_balance = row["Balance"] > median_balance

        if active and not low_product:
            return "Active Engaged"
        if active and low_product:
            return "Active, Low-Product"
        if (not active) and high_balance:
            return "Inactive, High-Balance"
        return "Inactive Disengaged"

    df["EngagementSegment"] = df.apply(classify, axis=1)
    return df


def product_utilization_features(df):
    """Step 3: Product Utilization Analysis features"""
    df = df.copy()
    df["IsSingleProduct"] = (df["NumOfProducts"] == 1).astype(int)
    df["IsMultiProduct"] = (df["NumOfProducts"] >= 2).astype(int)
    df["ProductDepthTier"] = pd.cut(
        df["NumOfProducts"], bins=[0, 1, 2, 4], labels=["Single", "Dual", "Multi (3-4)"]
    )
    return df


def financial_commitment_features(df):
    """Step 4: Financial Commitment vs Engagement Analysis"""
    df = df.copy()
    df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)
    median_balance = df["Balance"].median()
    df["IsHighBalance"] = (df["Balance"] > median_balance).astype(int)

    # Salary-balance mismatch: high salary but zero/very low balance suggests
    # funds are held elsewhere (weak relationship depth)
    df["SalaryBalanceRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    salary_median = df["EstimatedSalary"].median()
    df["SalaryBalanceMismatch"] = (
        (df["EstimatedSalary"] > salary_median) & (df["Balance"] == 0)
    ).astype(int)

    # "At-risk premium customer": high balance + inactive + few products
    df["AtRiskPremium"] = (
        (df["IsHighBalance"] == 1)
        & (df["IsActiveMember"] == 0)
        & (df["NumOfProducts"] <= 1)
    ).astype(int)

    return df


def retention_strength_features(df):
    """Step 5: Retention Strength Assessment
    Builds a composite 'Relationship Strength Index' and identifies
    'sticky customer' profiles.
    """
    df = df.copy()

    # Relationship Strength Index (0-4): activity + credit card + multi-product + tenure>=median
    tenure_median = df["Tenure"].median()
    df["RelationshipStrengthIndex"] = (
        df["IsActiveMember"]
        + df["HasCrCard"]
        + (df["NumOfProducts"] >= 2).astype(int)
        + (df["Tenure"] >= tenure_median).astype(int)
    )

    df["IsStickyCustomer"] = (df["RelationshipStrengthIndex"] >= 3).astype(int)

    return df


def build_full_feature_set(path):
    df = load_and_validate(path)
    df = engagement_classification(df)
    df = product_utilization_features(df)
    df = financial_commitment_features(df)
    df = retention_strength_features(df)
    return df


def compute_kpis(df):
    """Key Performance Indicators as specified in the brief."""
    kpis = {}

    # 1. Engagement Retention Ratio: churn rate active vs inactive
    active_churn = df.loc[df.IsActiveMember == 1, "Exited"].mean()
    inactive_churn = df.loc[df.IsActiveMember == 0, "Exited"].mean()
    kpis["Engagement Retention Ratio"] = {
        "Active churn rate": active_churn,
        "Inactive churn rate": inactive_churn,
        "Ratio (inactive/active)": inactive_churn / active_churn if active_churn > 0 else np.nan,
    }

    # 2. Product Depth Index: churn by number of products
    kpis["Product Depth Index"] = (
        df.groupby("NumOfProducts")["Exited"].mean().to_dict()
    )

    # 3. High-Balance Disengagement Rate
    high_bal_inactive = df[(df.IsHighBalance == 1) & (df.IsActiveMember == 0)]
    kpis["High-Balance Disengagement Rate"] = {
        "Churn rate (high-balance, inactive)": high_bal_inactive["Exited"].mean(),
        "Count": len(high_bal_inactive),
        "Share of high-balance customers": len(high_bal_inactive) / (df.IsHighBalance == 1).sum(),
    }

    # 4. Credit Card Stickiness Score
    cc_churn = df.loc[df.HasCrCard == 1, "Exited"].mean()
    no_cc_churn = df.loc[df.HasCrCard == 0, "Exited"].mean()
    kpis["Credit Card Stickiness Score"] = {
        "Churn rate (has card)": cc_churn,
        "Churn rate (no card)": no_cc_churn,
        "Difference": no_cc_churn - cc_churn,
    }

    # 5. Relationship Strength Index vs churn
    kpis["Relationship Strength Index"] = (
        df.groupby("RelationshipStrengthIndex")["Exited"].mean().to_dict()
    )

    return kpis


if __name__ == "__main__":
    df = build_full_feature_set("data/European_Bank.csv")
    print(df.head())
    print("\nEngagement segment distribution:")
    print(df["EngagementSegment"].value_counts())
    print("\nEngagement segment churn rates:")
    print(df.groupby("EngagementSegment")["Exited"].mean().sort_values(ascending=False))

    kpis = compute_kpis(df)
    import json
    print("\nKPIs:")
    print(json.dumps(kpis, indent=2, default=str))

    df.to_csv("data/European_Bank_engineered.csv", index=False)
