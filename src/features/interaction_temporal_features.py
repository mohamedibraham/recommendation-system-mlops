"""
Interaction Features & Temporal Features Computation
Handles user-item interaction matrix and time-based feature engineering.

Interaction Features:
  - Implicit feedback matrix (purchase + view + cart signals)
  - Composite interaction score per (user, product) pair
  - Session-level context
  - Negative feedback (returns)

Temporal Features:
  - User seasonality patterns (peak hours, days, months)
  - Product seasonality (monthly / quarterly sales)
  - Purchase cadence (inter-purchase interval)
"""

import logging

import pandas as pd

from src.utils.bq_utils import BigQueryManager
from src.queries.feature_queries import (
    InteractionFeatureQueries,
    TemporalFeatureQueries,
)

logger = logging.getLogger(__name__)



#                                                             Interaction Features
# _____________________________________________________________________________________________________________________________

class InteractionFeatureEngineer:
    """
    Builds the user-item interaction matrix and derived interaction signals.

    Why this matters for recommendations:
    - Pure collaborative filtering needs this matrix as input.
    - Implicit scores allow training without explicit ratings.
    - Negative feedback (returns) prevents recommending disliked items.
    """

    def __init__(self, bq: BigQueryManager) -> None:
        self.bq = bq

    def compute_interaction_matrix(self) -> pd.DataFrame:
        """
        Build the core user-item implicit feedback matrix.

        Returns a DataFrame with columns:
            user_id, product_id, purchase_count, view_count,
            cart_count, total_spend, implicit_score, last_purchased_at
        """
        logger.info("Computing user-item implicit feedback matrix …")
        df = self.bq.query_to_dataframe(
            InteractionFeatureQueries.USER_ITEM_IMPLICIT_FEEDBACK.value
        )

        # Normalise implicit_score to [0, 1]
        max_score = df["implicit_score"].max()
        if max_score > 0:
            df["implicit_score_normalised"] = df["implicit_score"] / max_score
        else:
            df["implicit_score_normalised"] = 0.0

        # Binary label for training (purchased ≥ 1 time = positive)
        df["label"] = (df["purchase_count"] > 0).astype(int)

        logger.info("Interaction matrix: %d (user, product) pairs.", len(df))
        return df

    def compute_negative_feedback(self) -> pd.DataFrame:
        """Returns DataFrame of (user_id, product_id) pairs the user returned."""
        logger.info("Computing negative feedback signals …")
        return self.bq.query_to_dataframe(
            InteractionFeatureQueries.NEGATIVE_FEEDBACK.value
        )

    def compute_session_features(self) -> pd.DataFrame:
        """Session-level features useful for sequential / session-based models."""
        logger.info("Computing session context features …")
        return self.bq.query_to_dataframe(
            InteractionFeatureQueries.SESSION_CONTEXT_FEATURES.value
        )

    def compute_all(self) -> dict:
        """
        Run all interaction feature computations.

        Returns:
            dict with keys: 'interaction_matrix', 'negative_feedback', 'sessions'
        """
        return {
            "interaction_matrix":  self.compute_interaction_matrix(),
            "negative_feedback":   self.compute_negative_feedback(),
            "sessions":            self.compute_session_features(),
        }



#                                                          Temporal Features
# ______________________________________________________________________________________________________________________

class TemporalFeatureEngineer:
    """
    Extracts time-based patterns from purchase and event history.

    Why this matters:
    - Users shop at specific times (morning coffee accessories, holiday gifts).
    - Products have seasonal demand patterns.
    - Recency decays: a purchase 2 years ago is less predictive than last week.
    - Purchase cadence reveals loyal vs lapsed users.
    """

    def __init__(self, bq: BigQueryManager) -> None:
        self.bq = bq

    def compute_user_seasonality(self) -> pd.DataFrame:
        """User-level temporal purchase patterns."""
        logger.info("Computing user purchase seasonality …")
        df = self.bq.query_to_dataframe(
            TemporalFeatureQueries.USER_PURCHASE_SEASONALITY.value
        )

        # Weekend vs weekday ratio
        total = (df["weekend_purchases"].fillna(0)
                 + df["weekday_purchases"].fillna(0))
        df["weekend_purchase_ratio"] = (
            df["weekend_purchases"].fillna(0) / total.replace(0, 1)
        )

        return df

    def compute_product_seasonality(self) -> pd.DataFrame:
        """
        Monthly/quarterly product demand.
        Pivoted so each product has 12 monthly sales columns.
        """
        logger.info("Computing product seasonality …")
        df = self.bq.query_to_dataframe(
            TemporalFeatureQueries.PRODUCT_SEASONALITY.value
        )

        # Pivot months into columns: product_id × month_sales_1 … month_sales_12
        monthly = df.pivot_table(
            index="product_id",
            columns="month",
            values="sales_count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        monthly.columns = (
            ["product_id"] + [f"sales_month_{int(m):02d}" for m in monthly.columns[1:]]
        )

        return monthly

    def compute_all(self) -> dict:
        return {
            "user_seasonality":    self.compute_user_seasonality(),
            "product_seasonality": self.compute_product_seasonality(),
        }