"""
User Features Computation
Computes all user-level features for the recommendation system.

Features produced:
  - Demographic features (age, gender, geography, acquisition channel)
  - Purchase behaviour (RFM: Recency, Frequency, Monetary)
  - Rolling window statistics (7/30/90/365 days)
  - Category & brand preferences (top-N)
  - Event / behavioural engagement signals
  - Derived conversion rates and purchase patterns
"""

import logging
from typing import Optional

import pandas as pd

from src.utils.bq_utils import BigQueryManager
from src.queries.feature_queries import UserFeatureQueries

logger = logging.getLogger(__name__)


class UserFeatureEngineer:
    """
    Computes and assembles all user-level features.

    Usage:
        eng = UserFeatureEngineer(bq_manager)
        df  = eng.compute_all_features()
    """

    def __init__(self, bq: BigQueryManager) -> None:
        self.bq = bq


    #                                                              Public API

    def compute_all_features(self) -> pd.DataFrame:
        """
        Run all user feature queries and join them into a single wide table.

        Returns:
            pd.DataFrame indexed by user_id containing all user features.
        """
        logger.info("Computing user core features …")
        core_df = self._get_core_features()

        logger.info("Computing user rolling window features …")
        rolling_df = self._get_rolling_features()

        logger.info("Computing user category preferences …")
        cat_df = self._get_category_preferences()

        logger.info("Computing user event / behavioural features …")
        event_df = self._get_event_features()

        # ── Join all feature groups on user_id 
        df = (
            core_df
            .merge(rolling_df, on="user_id", how="left")
            .merge(cat_df,     on="user_id", how="left")
            .merge(event_df,   on="user_id", how="left")
        )

        # ── Post-join derived features 
        df = self._add_derived_features(df)

        # ── Fill NaN for users with no history 
        df = self._fill_nulls(df)

        logger.info("User features computed: %d users × %d features.",
                    len(df), len(df.columns))
        return df

    #                                                   Private helpers
    
    def _get_core_features(self) -> pd.DataFrame:
        return self.bq.query_to_dataframe(
            UserFeatureQueries.USER_CORE_FEATURES.value
        )

    def _get_rolling_features(self) -> pd.DataFrame:
        return self.bq.query_to_dataframe(
            UserFeatureQueries.USER_ROLLING_FEATURES.value
        )

    def _get_category_preferences(self) -> pd.DataFrame:
        return self.bq.query_to_dataframe(
            UserFeatureQueries.USER_CATEGORY_PREFERENCES.value
        )

    def _get_event_features(self) -> pd.DataFrame:
        return self.bq.query_to_dataframe(
            UserFeatureQueries.USER_EVENT_FEATURES.value
        )

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features that require columns from multiple source tables.
        All these are Point-in-Time safe (no future leakage).
        """

        # ── RFM Scores (1-5 scale using quintiles)
        # Recency: lower days_since_last_order → better → score 5
        df["rfm_recency_score"] = pd.qcut(
            df["days_since_last_order"].fillna(9999),
            q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(float)

        # Frequency: higher total_orders → better → score 5
        df["rfm_frequency_score"] = pd.qcut(
            df["total_orders"].fillna(0).clip(lower=0),
            q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)

        # Monetary: higher total_spend → better → score 5
        df["rfm_monetary_score"] = pd.qcut(
            df["total_spend"].fillna(0).clip(lower=0),
            q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)

        # Composite RFM score
        df["rfm_composite_score"] = (
            df["rfm_recency_score"].fillna(1) * 0.4
            + df["rfm_frequency_score"].fillna(1) * 0.3
            + df["rfm_monetary_score"].fillna(1) * 0.3
        )

        # ── User Engagement Ratio 
        # How engaged is this user in browsing vs just purchasing?
        df["engagement_ratio"] = (
            df["total_events"].fillna(0)
            / df["total_orders"].replace(0, 1)
        )

        # ── Price Sensitivity 
        # Ratio of avg sale price paid vs avg retail price across their purchases
        # (requires product features merge downstream; placeholder computed here)
        df["price_tier"] = pd.cut(
            df["avg_item_price"].fillna(0),
            bins=[0, 25, 75, 150, float("inf")],
            labels=["budget", "mid", "premium", "luxury"],
        )

        # ── Account Activity Status 
        df["is_active_30d"] = (df["orders_last_30d"].fillna(0) > 0).astype(int)
        df["is_active_90d"] = (df["orders_last_90d"].fillna(0) > 0).astype(int)

        return df

    @staticmethod
    def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with sensible defaults per feature type."""
        fill_zero = [
            "total_orders", "total_items_purchased", "unique_products_purchased",
            "unique_categories_purchased", "unique_brands_purchased",
            "total_spend", "total_returns", "return_rate",
            "orders_last_7d", "orders_last_30d", "orders_last_90d", "orders_last_365d",
            "spend_last_7d", "spend_last_30d", "spend_last_90d", "spend_last_365d",
            "total_sessions", "total_events", "product_views", "cart_events",
            "purchase_events", "cancel_events", "view_to_purchase_rate",
            "cart_to_purchase_rate",
        ]
        fill_large = ["days_since_last_order", "days_since_last_event"]

        df[fill_zero]  = df[fill_zero].fillna(0)
        df[fill_large] = df[fill_large].fillna(9999)

        return df