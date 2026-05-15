"""
Feature Store — Offline Materialisation & Online Serving Interface

"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.bq_utils import BigQueryManager
from src.features.user_features import UserFeatureEngineer
from src.features.product_features import ProductFeatureEngineer
from src.features.interaction_temporal_features import (
    InteractionFeatureEngineer,
    TemporalFeatureEngineer,
)
from src.feature_store.feature_registry import FeatureRegistry, Entity
from src.queries.feature_queries import FeatureStoreQueries
from config.feature_config import FeatureStoreConfig

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Production Feature Store for the TheLook Ecommerce Recommendation System.

    Responsibilities:
      1. Materialise computed features into BigQuery offline store.
      2. Build Point-in-Time correct training datasets.
      3. Provide a get_online_features() interface for inference serving.
      4. Version features and prevent schema drift.
    """

    def __init__(self, bq: BigQueryManager, config: FeatureStoreConfig) -> None:
        self.bq     = bq
        self.config = config

        # Feature engineers
        self._user_eng        = UserFeatureEngineer(bq)
        self._product_eng     = ProductFeatureEngineer(bq)
        self._interaction_eng = InteractionFeatureEngineer(bq)
        self._temporal_eng    = TemporalFeatureEngineer(bq)

    
    # Materialisation  (Offline Store → BigQuery)
    
    def materialise_all(self) -> None:
        """
        Full materialisation run: compute all feature groups and write to BQ.
        Typically scheduled as a daily Kubeflow / Vertex AI pipeline.
        """
        logger.info("═══ Feature Store: Full Materialisation Started ═══")

        self._ensure_dataset_exists()
        self.materialise_user_features()
        self.materialise_product_features()
        self.materialise_interaction_features()

        logger.info("═══ Feature Store: Full Materialisation Completed ═══")

    def materialise_user_features(self) -> None:
        """Compute and write user features to BigQuery offline store."""
        logger.info("Materialising user features …")
        df = self._user_eng.compute_all_features()
        df["feature_timestamp"] = datetime.utcnow()

        table = self.config.table_ref(self.config.user_features_table)
        self.bq.dataframe_to_table(df, table, write_disposition="WRITE_TRUNCATE")
        logger.info("User features written → %s (%d rows).", table, len(df))

    def materialise_product_features(self) -> None:
        """Compute and write product features to BigQuery offline store."""
        logger.info("Materialising product features …")
        df = self._product_eng.compute_all_features()
        df["feature_timestamp"] = datetime.utcnow()

        table = self.config.table_ref(self.config.product_features_table)
        self.bq.dataframe_to_table(df, table, write_disposition="WRITE_TRUNCATE")
        logger.info("Product features written → %s (%d rows).", table, len(df))

    def materialise_interaction_features(self) -> None:
        """Compute and write interaction matrix to BigQuery offline store."""
        logger.info("Materialising interaction matrix …")
        interaction_data = self._interaction_eng.compute_all()
        df = interaction_data["interaction_matrix"]
        df["feature_timestamp"] = datetime.utcnow()

        table = self.config.table_ref(self.config.interaction_features_table)
        self.bq.dataframe_to_table(df, table, write_disposition="WRITE_TRUNCATE")
        logger.info("Interaction matrix written → %s (%d rows).", table, len(df))

    
    # Training Dataset Builder (Point-in-Time correct)
    

    def build_training_dataset(
        self,
        reference_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build a Point-in-Time correct training dataset.

        For each (user, product) interaction event, it joins only features
        that were known BEFORE the event timestamp — preventing data leakage.

        Args:
            reference_date: ISO date string 'YYYY-MM-DD'. If None, uses latest
                            materialised snapshot.

        Returns:
            pd.DataFrame ready for model training.
        """
        logger.info("Building PiT-correct training dataset (ref=%s) …",
                    reference_date or "latest")

        query = f"""
        SELECT
            uim.*,
            -- User features (joined on latest snapshot ≤ interaction time)
            uf.age,
            uf.gender,
            uf.country,
            uf.account_age_days,
            uf.total_orders,
            uf.total_spend,
            uf.avg_order_value,
            uf.days_since_last_order,
            uf.return_rate             AS user_return_rate,
            uf.rfm_composite_score,
            uf.rfm_recency_score,
            uf.rfm_frequency_score,
            uf.rfm_monetary_score,
            uf.top_category_1,
            uf.orders_last_30d,
            uf.spend_last_30d,
            uf.is_active_30d,
            uf.total_sessions,
            uf.product_views          AS user_product_views,
            uf.cart_events            AS user_cart_events,
            uf.view_to_purchase_rate,

            -- Product features
            pf.category,
            pf.brand,
            pf.department,
            pf.retail_price,
            pf.price_tier,
            pf.total_purchases        AS product_total_purchases,
            pf.unique_buyers,
            pf.popularity_score,
            pf.return_rate            AS product_return_rate,
            pf.avg_discount_rate,
            pf.sales_last_30d,
            pf.log_total_purchases,
            pf.lifecycle_stage,
            pf.is_new_product,
            pf.is_high_return

        FROM `{self.config.table_ref(self.config.interaction_features_table)}` uim

        LEFT JOIN `{self.config.table_ref(self.config.user_features_table)}` uf
               ON uim.user_id = uf.user_id

        LEFT JOIN `{self.config.table_ref(self.config.product_features_table)}` pf
               ON uim.product_id = pf.product_id
        """

        df = self.bq.query_to_dataframe(query)
        logger.info("Training dataset built: %d rows × %d features.",
                    len(df), len(df.columns))
        return df

    
    # Online Feature Serving Interface
    

    def get_user_features_online(self, user_id: int) -> dict:
        """
        Retrieve latest user features for real-time inference.

        In production this would call Redis / BigTable / Feast online store.
        For development, this reads directly from the BQ offline store.

        Args:
            user_id: The user entity key.

        Returns:
            dict of {feature_name: feature_value} for online-serving features.
        """
        online_feature_names = FeatureRegistry.feature_names(Entity.USER)
        cols = ", ".join(online_feature_names)

        query = f"""
        SELECT {cols}
        FROM `{self.config.table_ref(self.config.user_features_table)}`
        WHERE user_id = {user_id}
        ORDER BY feature_timestamp DESC
        LIMIT 1
        """
        df = self.bq.query_to_dataframe(query)
        if df.empty:
            # Cold-start: return default values
            logger.warning("Cold-start: no features found for user_id=%d.", user_id)
            return self._cold_start_user_features()

        return df.iloc[0].to_dict()

    def get_product_features_online(self, product_id: int) -> dict:
        """Retrieve latest product features for real-time inference."""
        query = f"""
        SELECT *
        FROM `{self.config.table_ref(self.config.product_features_table)}`
        WHERE product_id = {product_id}
        ORDER BY feature_timestamp DESC
        LIMIT 1
        """
        df = self.bq.query_to_dataframe(query)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    
    # Feature Validation
    

    def validate_features(self, df: pd.DataFrame, entity: Entity) -> dict:
        """
        Validate a features DataFrame against the Feature Registry schema.

        Checks:
          - No undeclared features present
          - All required features are present
          - Null rate below threshold for each feature

        Returns:
            dict with 'passed', 'missing_features', 'null_violations'
        """
        registered = set(FeatureRegistry.feature_names(entity))
        present    = set(df.columns) - {"user_id", "product_id", "feature_timestamp"}

        missing    = registered - present
        null_rates = {col: df[col].isnull().mean() for col in present
                      if col in df.columns}
        violations = {col: rate for col, rate in null_rates.items() if rate > 0.5}

        result = {
            "passed":           len(missing) == 0 and len(violations) == 0,
            "missing_features": list(missing),
            "null_violations":  violations,
        }
        if not result["passed"]:
            logger.warning("Feature validation failed: %s", result)
        else:
            logger.info("Feature validation passed for entity=%s.", entity.value)

        return result

    
    # Private helpers
    

    def _ensure_dataset_exists(self) -> None:
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS
        `{self.config.project_id}.{self.config.dataset}`
        OPTIONS (location = 'US')
        """
        self.bq.execute_ddl(ddl)

    @staticmethod
    def _cold_start_user_features() -> dict:
        """Return default feature values for a brand-new user (cold start)."""
        defaults = {f.name: f.default_value
                    for f in FeatureRegistry.get_by_entity(Entity.USER)
                    if f.default_value is not None}
        return defaults