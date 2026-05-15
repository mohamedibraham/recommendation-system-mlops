"""
Product Features Computation
Computes all product-level features for the recommendation system.

Features produced:
  - Static catalogue features (category, brand, department, price, margin)
  - Popularity metrics (total purchases, unique buyers, revenue)
  - Quality signals (return rate, discount depth)
  - Temporal popularity (rolling 7/30/90d sales + trend)
  - Co-purchase diversity (how often bought with other products)
  - Normalised popularity & reach scores (within-category)
"""

import logging

import pandas as pd

from src.utils.bq_utils import BigQueryManager
from src.queries.feature_queries import ProductFeatureQueries

logger = logging.getLogger(__name__)


class ProductFeatureEngineer:
    """
    Computes and assembles all product-level features.

    Usage:
        eng = ProductFeatureEngineer(bq_manager)
        df  = eng.compute_all_features()
    """

    def __init__(self, bq: BigQueryManager) -> None:
        self.bq = bq

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_features(self) -> pd.DataFrame:
        """
        Run all product feature queries and join them into a single table.

        Returns:
            pd.DataFrame indexed by product_id with all product features.
        """
        logger.info("Computing product core features …")
        core_df = self.bq.query_to_dataframe(
            ProductFeatureQueries.PRODUCT_CORE_FEATURES.value
        )

        logger.info("Computing product popularity scores …")
        pop_df = self.bq.query_to_dataframe(
            ProductFeatureQueries.PRODUCT_POPULARITY_SCORE.value
        )

        logger.info("Computing product rolling sales …")
        rolling_df = self.bq.query_to_dataframe(
            ProductFeatureQueries.PRODUCT_ROLLING_SALES.value
        )

        logger.info("Computing co-purchase stats …")
        copurchase_df = self.bq.query_to_dataframe(
            ProductFeatureQueries.PRODUCT_CO_PURCHASE_STATS.value
        )

        # ── Join all feature groups ───────────────────────────────────
        df = (
            core_df
            .merge(pop_df[["product_id", "popularity_score", "reach_score"]],
                   on="product_id", how="left")
            .merge(rolling_df,    on="product_id", how="left")
            .merge(copurchase_df, on="product_id", how="left")
        )

        # ── Post-join derived features ────────────────────────────────
        df = self._add_derived_features(df)
        df = self._fill_nulls(df)

        logger.info("Product features computed: %d products × %d features.",
                    len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that are computed from multiple base columns."""

        # ── Price Tier Encoding ───────────────────────────────────────
        df["price_tier"] = pd.cut(
            df["retail_price"].fillna(0),
            bins=[0, 25, 75, 150, float("inf")],
            labels=["budget", "mid", "premium", "luxury"],
        )

        # ── Product Lifecycle Stage ───────────────────────────────────
        # Based on rolling sales momentum
        rolling_30  = df["sales_last_30d"].fillna(0)
        rolling_90  = df["sales_last_90d"].fillna(0)
        trend       = df["sales_trend_30_vs_prev"].fillna(0)

        def lifecycle_stage(row) -> str:
            if row["total_purchases"] == 0:
                return "no_sales"
            if row["sales_last_30d"] == 0:
                return "declining"
            if row.get("sales_trend_30_vs_prev", 0) > 1.5:
                return "growing"
            if row.get("sales_trend_30_vs_prev", 0) < 0.5:
                return "declining"
            return "stable"

        df["lifecycle_stage"] = df.apply(lifecycle_stage, axis=1)

        # ── Log-transformed popularity (for embedding inputs) ─────────
        import numpy as np
        df["log_total_purchases"] = np.log1p(df["total_purchases"].fillna(0))
        df["log_unique_buyers"]   = np.log1p(df["unique_buyers"].fillna(0))
        df["log_total_revenue"]   = np.log1p(df["total_revenue"].fillna(0))

        # ── Novelty: is this a new product (< 30 days old in catalogue)? ─
        df["is_new_product"] = (
            df["active_days"].fillna(0) < 30
        ).astype(int)

        # ── High-return flag ──────────────────────────────────────────
        df["is_high_return"] = (df["return_rate"].fillna(0) > 0.15).astype(int)

        return df

    @staticmethod
    def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
        fill_zero = [
            "total_purchases", "unique_buyers", "total_revenue",
            "return_rate", "avg_discount_rate",
            "sales_last_7d", "sales_last_30d", "sales_last_90d",
            "orders_with_co_purchase", "unique_co_purchased_products",
            "popularity_score", "reach_score",
        ]
        df[fill_zero] = df[fill_zero].fillna(0)
        return df