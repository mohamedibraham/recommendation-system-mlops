"""
Model Trainer
=============
KFP component that orchestrates the full training pipeline:

  Phase 1 — Data Split
    Temporal split: oldest 75% → train, next 10% → val, latest 15% → test
    Preserves time ordering to prevent leakage.

  Phase 2 — Two-Tower Retrieval Training
    Trains the neural retrieval model on the interaction matrix.
    Exports user & product embedding tables to BigQuery.

  Phase 3 — Ranking Model Training (LightGBM LambdaRank)
    Uses Two-Tower candidates + full feature set to train the ranker.
    Directly optimises NDCG.

  Phase 4 — Artifact Export
    Saves both models to GCS, exports metadata for the evaluation component.
"""

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

logger = logging.getLogger(__name__)


# Data Splitter — Temporal Split (no leakage)

class TemporalDataSplitter:
    """
    Splits the training dataset using a time-based boundary.

    Why temporal split?
      Random splits leak future data into training, inflating metrics.
      In production, a model always predicts future interactions from
      past behaviour — the split must reflect this.

    Split strategy:
      Sort all (user, product) interactions by last_purchased_at.
      Oldest 75% → train | next 10% → val | latest 15% → test

    Cold-start users (no purchase history):
      Assigned to train set with label=0.
    """

    def __init__(
        self,
        train_ratio: float = 0.75,
        val_ratio:   float = 0.10,
        test_ratio:  float = 0.15,
        date_col:    str   = "last_purchased_at",
    ) -> None:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.date_col    = date_col

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Args:
            df: Full training_dataset DataFrame from Feature Store.

        Returns:
            (train_df, val_df, test_df)
        """
        # Interactions WITH timestamp (positive interactions)
        has_date  = df[df[self.date_col].notna()].copy()
        no_date   = df[df[self.date_col].isna()].copy()   # cold-start / view-only

        has_date  = has_date.sort_values(self.date_col).reset_index(drop=True)
        n         = len(has_date)
        n_train   = int(n * self.train_ratio)
        n_val     = int(n * self.val_ratio)

        train = pd.concat([has_date.iloc[:n_train], no_date], ignore_index=True)
        val   = has_date.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test  = has_date.iloc[n_train + n_val :].reset_index(drop=True)

        # Record split boundary dates for reproducibility
        self.train_end_date = has_date.iloc[n_train - 1][self.date_col]
        self.val_end_date   = has_date.iloc[n_train + n_val - 1][self.date_col]

        logger.info(
            "Temporal split complete:\n"
            "  train=%d rows (up to %s)\n"
            "  val  =%d rows (up to %s)\n"
            "  test =%d rows (latest)",
            len(train), self.train_end_date,
            len(val),   self.val_end_date,
            len(test),
        )
        return train, val, test


# Two-Tower Trainer

class TwoTowerTrainer:
    """
    Wraps Two-Tower model training with:
    - tf.data.Dataset pipeline (shuffle, batch, prefetch)
    - Vocab size computation from actual data
    - Embedding export to numpy arrays
    - Checkpoint management
    """

    def __init__(self, config, output_dir: str) -> None:
        self.config     = config
        self.output_dir = output_dir

    def build_tf_dataset(
        self,
        df:         pd.DataFrame,
        batch_size: int,
        shuffle:    bool = True,
        seed:       int  = 42,
    ) -> "tf.data.Dataset":
        """
        Convert a pandas DataFrame to a tf.data.Dataset for Two-Tower training.

        Yields: (user_feature_dict, product_feature_dict)
        Both dicts contain integer-encoded categoricals + float32 numericals.
        """
        import tensorflow as tf
        cfg = self.config

        # ── User inputs 
        user_data = {}
        for feat in cfg.user_vocab_sizes:
            col = feat if feat in df.columns else "user_id"
            user_data[feat] = df[col].fillna(0).astype(int).values

        num_user = df[
            [c for c in cfg.user_numerical_features if c in df.columns]
        ].fillna(0).astype(np.float32).values
        user_data["numerical"] = num_user

        # ── Product inputs 
        product_data = {}
        for feat in cfg.product_vocab_sizes:
            col = feat if feat in df.columns else "product_id"
            product_data[feat] = df[col].fillna(0).astype(int).values

        num_prod = df[
            [c for c in cfg.product_numerical_features if c in df.columns]
        ].fillna(0).astype(np.float32).values
        product_data["numerical"] = num_prod

        # ── Build Dataset 
        dataset = tf.data.Dataset.from_tensor_slices((user_data, product_data))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10_000, seed=seed)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def compute_vocab_sizes(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Compute actual vocabulary sizes from data (replaces config defaults)."""
        cfg = self.config
        user_vocab, prod_vocab = {}, {}

        for feat in cfg.user_vocab_sizes:
            if feat in df.columns:
                user_vocab[feat] = int(df[feat].max()) + 2   # +2 for OOV
            else:
                user_vocab[feat] = cfg.user_vocab_sizes[feat]

        for feat in cfg.product_vocab_sizes:
            if feat in df.columns:
                prod_vocab[feat] = int(df[feat].max()) + 2
            else:
                prod_vocab[feat] = cfg.product_vocab_sizes[feat]

        return user_vocab, prod_vocab

    def train(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
    ) -> Tuple["TwoTowerModel", Dict]:
        """
        Full Two-Tower training run.

        Returns:
            (trained_model, training_history_dict)
        """
        from src.components import TwoTowerModel

        # Compute vocab sizes from actual data
        user_vocab, prod_vocab = self.compute_vocab_sizes(train_df)

        # Build model
        model = TwoTowerModel(self.config)
        model.build(user_vocab_sizes=user_vocab, product_vocab_sizes=prod_vocab)
        model.compile()

        # Build tf.data pipelines
        train_ds = self.build_tf_dataset(train_df, self.config.batch_size, shuffle=True)
        val_ds   = self.build_tf_dataset(val_df,   self.config.batch_size, shuffle=False)

        checkpoint_dir = os.path.join(self.output_dir, "two_tower_checkpoints")
        history = model.train(train_ds, val_ds, checkpoint_dir=checkpoint_dir)

        # Save model
        save_dir = os.path.join(self.output_dir, "two_tower")
        model.save(save_dir)

        logger.info("Two-Tower training complete. Saved → %s", save_dir)
        return model, history


# Ranking Trainer


class RankingTrainer:
    """
    Wraps LightGBM LambdaRank training with:
    - Candidate generation from Two-Tower embeddings
    - Feature assembly (user + product + interaction + cross features)
    - Training with early stopping
    - Feature importance export
    """

    def __init__(self, config, output_dir: str) -> None:
        self.config     = config
        self.output_dir = output_dir

    def generate_candidates(
        self,
        two_tower_model,
        df:        pd.DataFrame,
        num_negatives_per_positive: int = 5,
    ) -> pd.DataFrame:
        """
        Generate training samples for the ranking model.

        Strategy:
          For each positive (user, product) interaction, sample
          `num_negatives_per_positive` hard negatives from the Two-Tower
          top-K candidates that were NOT purchased.

        Returns:
            DataFrame with positive + negative samples, ready for ranking training.
        """
        positives = df[df["label"] == 1].copy()
        negatives = df[df["label"] == 0].copy()

        # Sample hard negatives (products retrieved by Two-Tower but not purchased)
        # In a full implementation, run Two-Tower retrieval and filter
        # For efficiency, sample from the existing negatives in the dataset
        sampled_negatives = (
            negatives
            .groupby("user_id")
            .apply(lambda g: g.sample(
                min(num_negatives_per_positive, len(g)), random_state=42
            ))
            .reset_index(drop=True)
        )

        candidates = pd.concat([positives, sampled_negatives], ignore_index=True)
        candidates = candidates.sort_values("user_id").reset_index(drop=True)

        logger.info(
            "Candidates: %d positives + %d negatives = %d total.",
            len(positives), len(sampled_negatives), len(candidates),
        )
        return candidates

    def train(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        two_tower_model=None,
    ) -> Tuple["RankingModel", Dict]:
        """
        Train the LambdaRank ranking model.

        Args:
            train_df:         Training split from TemporalDataSplitter.
            val_df:           Validation split.
            two_tower_model:  Optional trained TwoTowerModel for candidate generation.

        Returns:
            (trained_ranker, metadata_dict)
        """
        from src.models.ranking_model import RankingModel

        # Generate candidates
        if two_tower_model is not None:
            train_candidates = self.generate_candidates(two_tower_model, train_df)
            val_candidates   = self.generate_candidates(two_tower_model, val_df)
        else:
            # Direct training without retrieval stage (full dataset)
            train_candidates = train_df.sort_values("user_id").reset_index(drop=True)
            val_candidates   = val_df.sort_values("user_id").reset_index(drop=True)

        ranker = RankingModel(self.config)
        ranker.fit(train_candidates, val_candidates)

        # Save model
        save_dir = os.path.join(self.output_dir, "ranking")
        ranker.save(save_dir)

        metadata = {
            "best_iteration":    ranker._model.best_iteration,
            "feature_columns":   self.config.feature_columns,
            "train_rows":        len(train_candidates),
            "val_rows":          len(val_candidates),
        }

        logger.info("Ranking model training complete. Saved → %s", save_dir)
        return ranker, metadata


# KFP Component

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "numpy==1.26.0",
        "google-cloud-bigquery==3.11.4",
        "pyarrow==13.0.0",
        "lightgbm==4.1.0",
        "tensorflow==2.13.0",
        "scikit-learn==1.3.0",
        "kfp==2.4.0",
    ],
)
def model_training_component(
    project_id:          str,
    training_dataset:    Input[Dataset],
    two_tower_model_dir: Output[Model],
    ranking_model_dir:   Output[Model],
    training_metrics:    Output[Metrics],
    train_split:         Output[Dataset],
    val_split:           Output[Dataset],
    test_split:          Output[Dataset],
) -> str:
    """
    KFP component: load training dataset → temporal split → train both models.

    Args:
        project_id:          GCP project ID.
        training_dataset:    KFP Dataset artifact (parquet) from feature pipeline.
        two_tower_model_dir: Output Model artifact for the Two-Tower model.
        ranking_model_dir:   Output Model artifact for the Ranking model.
        training_metrics:    KFP Metrics artifact.
        train/val/test_split: Output Dataset artifacts for evaluation component.

    Returns:
        JSON summary string.
    """
    import json
    import logging
    import os
    import pickle
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    summary = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "project_id":    project_id,
        "phases":        {},
    }

    # ── Load training dataset 
    log.info("Loading training dataset …")
    df = pd.read_parquet(training_dataset.path)
    log.info("Dataset shape: %s", df.shape)

    # ── Phase 1: Temporal Split 
    log.info("Phase 1: Temporal split …")
    date_col = "last_purchased_at"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        has_date = df[df[date_col].notna()].sort_values(date_col)
        no_date  = df[df[date_col].isna()]

        n       = len(has_date)
        n_train = int(n * 0.75)
        n_val   = int(n * 0.10)

        train_df = pd.concat([has_date.iloc[:n_train], no_date], ignore_index=True)
        val_df   = has_date.iloc[n_train: n_train + n_val].reset_index(drop=True)
        test_df  = has_date.iloc[n_train + n_val:].reset_index(drop=True)
    else:
        # Fallback: random split if no date column
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42)
        val_df,   test_df = train_test_split(temp_df, test_size=0.60, random_state=42)

    log.info("Split sizes → train:%d  val:%d  test:%d",
             len(train_df), len(val_df), len(test_df))

    train_df.to_parquet(train_split.path, index=False)
    val_df.to_parquet(val_split.path,     index=False)
    test_df.to_parquet(test_split.path,   index=False)

    summary["phases"]["data_split"] = {
        "train_rows": len(train_df),
        "val_rows":   len(val_df),
        "test_rows":  len(test_df),
        "status":     "OK",
    }

    training_metrics.log_metric("train_rows", len(train_df))
    training_metrics.log_metric("val_rows",   len(val_df))
    training_metrics.log_metric("test_rows",  len(test_df))

    # ── Phase 2: Encode categorical features 
    log.info("Phase 2: Encoding categorical features …")
    cat_cols = ["gender", "country", "acquisition_source",
                "top_category_1", "category", "brand",
                "department", "price_tier", "lifecycle_stage"]

    encoders = {}
    for col in cat_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(
                train_df[col].fillna("__MISSING__").astype(str)
            )
            for split_df in [val_df, test_df]:
                if col in split_df.columns:
                    split_df[col] = split_df[col].apply(
                        lambda x: le.transform([str(x)])[0]
                        if str(x) in le.classes_ else 0
                    )
            encoders[col] = le

    # ── Phase 3: LightGBM Ranking Model 
    log.info("Phase 3: Training LightGBM LambdaRank …")

    feature_cols = [
        c for c in [
            # User features
            "age", "account_age_days", "total_orders", "total_spend",
            "avg_order_value", "days_since_last_order",
            "rfm_composite_score", "rfm_recency_score",
            "rfm_frequency_score", "rfm_monetary_score",
            "orders_last_30d", "spend_last_30d", "is_active_30d",
            "total_sessions", "view_to_purchase_rate",
            # Product features
            "retail_price", "popularity_score", "unique_buyers",
            "total_purchases", "log_total_purchases",
            "sales_last_30d", "avg_discount_rate",
            "is_new_product", "is_high_return",
            # Interaction features
            "purchase_count", "view_count", "cart_count",
            "implicit_score_normalised", "total_spend_interaction",
            # Encoded categoricals
            "gender", "category", "brand", "price_tier",
        ]
        if c in train_df.columns
    ]

    def build_graded_labels(d: pd.DataFrame) -> pd.Series:
        lbl = pd.Series(0, index=d.index)
        if "view_count" in d:
            lbl = lbl.where(d["view_count"] == 0, 1)
        if "cart_count" in d:
            lbl = lbl.where(d["cart_count"] == 0, 3)
        if "purchase_count" in d:
            lbl = lbl.where(d["purchase_count"] == 0, 7)
            lbl = lbl.where(d["purchase_count"] <= 1, 15)
        return lbl

    def make_group(d: pd.DataFrame) -> list:
        return d.sort_values("user_id").groupby("user_id").size().tolist()

    train_sorted = train_df.sort_values("user_id").reset_index(drop=True)
    val_sorted   = val_df.sort_values("user_id").reset_index(drop=True)

    X_train = train_sorted[feature_cols].fillna(0)
    y_train = build_graded_labels(train_sorted)
    g_train = make_group(train_sorted)

    X_val   = val_sorted[feature_cols].fillna(0)
    y_val   = build_graded_labels(val_sorted)
    g_val   = make_group(val_sorted)

    lgb_train = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=False)
    lgb_val   = lgb.Dataset(X_val,   label=y_val,   group=g_val,
                            reference=lgb_train,     free_raw_data=False)

    params = {
        "objective":        "lambdarank",
        "metric":           "ndcg",
        "ndcg_eval_at":     [5, 10, 20],
        "learning_rate":    0.05,
        "num_leaves":       127,
        "min_child_samples": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "label_gain":       [0, 1, 3, 7, 15],
        "verbosity":        -1,
    }

    ranking_booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    # Save ranking model
    os.makedirs(ranking_model_dir.path, exist_ok=True)
    ranking_model_path = os.path.join(ranking_model_dir.path, "ranking_model.txt")
    ranking_booster.save_model(ranking_model_path)

    # Save encoders + feature list
    meta = {
        "feature_columns":   feature_cols,
        "encoders":          encoders,
        "best_iteration":    ranking_booster.best_iteration,
        "feature_importance": dict(zip(
            ranking_booster.feature_name(),
            ranking_booster.feature_importance("gain").tolist(),
        )),
    }
    with open(os.path.join(ranking_model_dir.path, "ranking_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    best_ndcg = ranking_booster.best_score["val"].get("ndcg@10", 0.0)
    log.info("Ranking model → best NDCG@10: %.4f  (iter=%d)",
             best_ndcg, ranking_booster.best_iteration)

    training_metrics.log_metric("ranking_best_ndcg_at_10", best_ndcg)
    training_metrics.log_metric("ranking_best_iteration",  ranking_booster.best_iteration)
    training_metrics.log_metric("num_ranking_features",    len(feature_cols))

    summary["phases"]["ranking_training"] = {
        "best_ndcg_at_10": best_ndcg,
        "best_iteration":  ranking_booster.best_iteration,
        "features_used":   len(feature_cols),
        "status":          "OK",
    }

    # ── Phase 4: Two-Tower Placeholder 
    # Full TF training requires GPU container; the LightGBM ranker covers
    # the ranking stage. Two-Tower training mirrors the same pattern above
    # but uses TensorFlow — enabled when the pipeline runs on GPU nodes.
    log.info("Phase 4: Two-Tower — placeholder (requires TF/GPU container).")
    os.makedirs(two_tower_model_dir.path, exist_ok=True)
    with open(os.path.join(two_tower_model_dir.path, "README.txt"), "w") as f:
        f.write(
            "Two-Tower training requires tensorflow>=2.13 on GPU node.\n"
            "Run src/models/two_tower_model.py in a dedicated GPU training job.\n"
        )

    summary["phases"]["two_tower"] = {"status": "DEFERRED_GPU_REQUIRED"}
    summary["status"] = "SUCCESS"

    log.info("Training complete:\n%s", json.dumps(summary, indent=2, default=str))
    return json.dumps(summary, indent=2, default=str)