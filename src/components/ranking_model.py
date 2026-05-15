"""
LightGBM LambdaRank — Ranking Stage
=====================================
Takes the ~100 candidates produced by the Two-Tower retrieval stage
and re-ranks them using a gradient boosted tree with LambdaRank objective.

Why LightGBM for ranking?
  - Directly optimises NDCG (LambdaRank loss)
  - Handles heterogeneous features (categorical + numerical) natively
  - Fast training and inference
  - Interpretable via SHAP feature importance
  - No normalisation / embedding needed for tabular cross-features

Input features (from training_dataset, per candidate):
  - User features    (RFM, rolling windows, demographics)
  - Product features (popularity, price, lifecycle)
  - Interaction features (implicit_score, view/cart/purchase counts)
  - Cross features   (computed in _add_cross_features)

Label:
  - Graded relevance: 0 (no interaction) → 1 (view) → 3 (cart) → 7 (purchase)
  - Binary label for simpler setup: purchased = 1, else = 0
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not installed. RankingModel will be unavailable.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# Graded Label Builder
# ════════════════════════════════════════════════════════════════════════════

def build_graded_labels(df: pd.DataFrame) -> pd.Series:
    """
    Convert raw interaction counts to graded relevance labels.

    Scale (aligned with LightGBM label_gain config):
      0  → no interaction
      1  → viewed (view_count > 0, no cart/purchase)
      3  → added to cart (cart_count > 0, no purchase)
      7  → purchased once
      15 → purchased 2+ times (high loyalty)
    """
    labels = pd.Series(0, index=df.index)
    labels = labels.where(df.get("view_count",     pd.Series(0, index=df.index)) == 0, 1)
    labels = labels.where(df.get("cart_count",     pd.Series(0, index=df.index)) == 0, 3)
    labels = labels.where(df.get("purchase_count", pd.Series(0, index=df.index)) == 0, 7)
    labels = labels.where(df.get("purchase_count", pd.Series(0, index=df.index)) <= 1, 15)
    return labels


# ════════════════════════════════════════════════════════════════════════════
# RankingModel
# ════════════════════════════════════════════════════════════════════════════

class RankingModel:
    """
    Production LightGBM LambdaRank model for the ranking stage.

    Usage:
        ranker = RankingModel(config)
        ranker.fit(train_df, val_df)
        ranked_df = ranker.predict(candidates_df)
        ranker.save("/gcs/models/ranker_v1")
    """

    def __init__(self, config) -> None:
        if not LGB_AVAILABLE:
            raise RuntimeError("LightGBM is required for RankingModel.")
        self.config   = config
        self._model   : Optional["lgb.Booster"] = None
        self._feature_importance : Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────
    # Data Preparation
    # ──────────────────────────────────────────────────────────────────────

    def prepare_dataset(
        self,
        df:           pd.DataFrame,
        graded_labels: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare a DataFrame for LightGBM LambdaRank training.

        Args:
            df:            Full training_dataset DataFrame (from Feature Store)
            graded_labels: Use graded relevance (0/1/3/7/15) vs binary (0/1)

        Returns:
            (X, y, groups) where groups is the query-group size array
            required by LambdaRank.
        """
        cfg = self.config

        # ── Cross features ────────────────────────────────────────────────
        df = self._add_cross_features(df.copy())

        # ── Feature matrix ────────────────────────────────────────────────
        available = [c for c in cfg.feature_columns if c in df.columns]
        missing   = set(cfg.feature_columns) - set(available)
        if missing:
            logger.warning("Missing feature columns (will be skipped): %s", missing)
        X = df[available]

        # ── Labels ───────────────────────────────────────────────────────
        y = build_graded_labels(df) if graded_labels else df[cfg.label_column]

        # ── Group array (required for LambdaRank) ─────────────────────────
        # Sort by user_id first (LGB requires consecutive groups)
        groups = df.groupby(cfg.group_column).size().reset_index(drop=True)

        return X, y, groups

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
    ) -> "lgb.Booster":
        """
        Train the LambdaRank model.

        Args:
            train_df: Training DataFrame (sorted by user_id).
            val_df:   Validation DataFrame (sorted by user_id).

        Returns:
            Trained lgb.Booster instance.
        """
        cfg = self.config

        logger.info("Preparing training data …")
        X_train, y_train, g_train = self.prepare_dataset(train_df)
        X_val,   y_val,   g_val   = self.prepare_dataset(val_df)

        logger.info(
            "Training set: %d rows, %d features, %d query groups",
            len(X_train), len(X_train.columns), len(g_train)
        )

        # ── LightGBM Datasets ─────────────────────────────────────────────
        lgb_train = lgb.Dataset(
            X_train, label=y_train,
            group=g_train,
            free_raw_data=False,
        )
        lgb_val = lgb.Dataset(
            X_val, label=y_val,
            group=g_val,
            reference=lgb_train,
            free_raw_data=False,
        )

        # ── Hyperparameters ───────────────────────────────────────────────
        params = {
            "objective":           cfg.objective,
            "metric":              cfg.metric,
            "ndcg_eval_at":        cfg.ndcg_eval_at,
            "learning_rate":       cfg.learning_rate,
            "num_leaves":          cfg.num_leaves,
            "max_depth":           cfg.max_depth,
            "min_child_samples":   cfg.min_child_samples,
            "subsample":           cfg.subsample,
            "colsample_bytree":    cfg.colsample_bytree,
            "reg_alpha":           cfg.reg_alpha,
            "reg_lambda":          cfg.reg_lambda,
            "label_gain":          cfg.label_gain,
            "verbosity":           cfg.verbosity,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        logger.info("Training LightGBM LambdaRank …")
        self._model = lgb.train(
            params,
            lgb_train,
            num_boost_round=cfg.n_estimators,
            valid_sets=[lgb_val],
            valid_names=["val"],
            callbacks=callbacks,
        )

        # ── Feature importance ────────────────────────────────────────────
        self._feature_importance = pd.DataFrame({
            "feature":    self._model.feature_name(),
            "importance_gain":  self._model.feature_importance("gain"),
            "importance_split": self._model.feature_importance("split"),
        }).sort_values("importance_gain", ascending=False)

        logger.info("Training complete. Best iteration: %d",
                    self._model.best_iteration)
        logger.info("\nTop-10 features by gain:\n%s",
                    self._feature_importance.head(10).to_string(index=False))

        return self._model

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score and rank candidates for each user.

        Args:
            candidates_df: DataFrame with columns matching feature_columns.
                           Typically the output of Two-Tower retrieval.

        Returns:
            DataFrame with additional 'ranking_score' column, sorted by
            (user_id, ranking_score DESC).
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        df = self._add_cross_features(candidates_df.copy())
        available = [c for c in self.config.feature_columns if c in df.columns]
        X = df[available]

        scores = self._model.predict(X, num_iteration=self._model.best_iteration)
        df["ranking_score"] = scores

        ranked = (
            df.sort_values(["user_id", "ranking_score"], ascending=[True, False])
              .reset_index(drop=True)
        )
        return ranked

    def predict_top_k(
        self,
        candidates_df: pd.DataFrame,
        k:             int = 10,
    ) -> pd.DataFrame:
        """
        Return top-k ranked products per user.

        Args:
            candidates_df: Output of Two-Tower retrieval.
            k:             Number of recommendations per user.

        Returns:
            DataFrame with top-k products per user_id.
        """
        ranked = self.predict(candidates_df)
        top_k  = (
            ranked.groupby("user_id")
                  .head(k)
                  .reset_index(drop=True)
        )
        return top_k

    # ──────────────────────────────────────────────────────────────────────
    # SHAP Explainability
    # ──────────────────────────────────────────────────────────────────────

    def explain(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Compute SHAP values for interpretability.

        Returns:
            SHAP values array (n_samples × n_features) or None if shap unavailable.
        """
        if not SHAP_AVAILABLE:
            logger.warning("shap not installed. Skipping explainability.")
            return None
        explainer   = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(X)
        return shap_values

    # ──────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """Save model to disk (LightGBM .txt + pickle for metadata)."""
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "ranking_model.txt")
        meta_path  = os.path.join(output_dir, "ranking_meta.pkl")

        self._model.save_model(model_path)
        with open(meta_path, "wb") as f:
            pickle.dump({
                "feature_importance": self._feature_importance,
                "feature_columns":    self.config.feature_columns,
                "best_iteration":     self._model.best_iteration,
            }, f)

        logger.info("RankingModel saved → %s", output_dir)

    @classmethod
    def load(cls, model_dir: str, config) -> "RankingModel":
        """Load a saved RankingModel."""
        instance = cls(config)
        model_path = os.path.join(model_dir, "ranking_model.txt")
        instance._model = lgb.Booster(model_file=model_path)
        meta_path = os.path.join(model_dir, "ranking_meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            instance._feature_importance = meta.get("feature_importance")
        logger.info("RankingModel loaded from %s", model_dir)
        return instance

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross features that capture user × product interactions.
        These are key for ranking and cannot be precomputed per entity alone.
        """
        # Price affinity: does the product price match the user's avg spending?
        if "retail_price" in df and "avg_item_price_u" in df:
            df["price_affinity"] = 1.0 - (
                (df["retail_price"] - df["avg_item_price_u"]).abs()
                / (df["avg_item_price_u"].replace(0, 1))
            ).clip(0, 1)

        # Category match: did the user ever buy from this category?
        if "category" in df and "top_category_1_u" in df:
            df["category_is_top1"] = (
                df["category"] == df["top_category_1_u"]
            ).astype(int)

        # Recency × popularity interaction
        if "days_since_last_order_u" in df and "popularity_score" in df:
            df["recency_x_popularity"] = (
                (1.0 / (1.0 + df["days_since_last_order_u"].fillna(9999)))
                * df["popularity_score"].fillna(0)
            )

        # User spend tier alignment with product price tier
        if "rfm_monetary_score_u" in df and "retail_price" in df:
            df["spend_price_alignment"] = (
                df["rfm_monetary_score_u"].fillna(1) *
                np.log1p(df["retail_price"].fillna(0))
            )

        return df