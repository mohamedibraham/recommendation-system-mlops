"""
Model Configuration
All hyperparameters, thresholds, and training settings for the
Two-Tower Retrieval model and LightGBM Ranking model.

Designed for easy override via Vertex AI custom training job parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Two-Tower Model Config 

@dataclass
class TowerConfig:
    """Architecture config for one tower (user or product)."""
    embedding_dim:      int        = 64       # entity embedding size
    hidden_layers:      List[int]  = field(default_factory=lambda: [256, 128, 64])
    dropout_rate:       float      = 0.3
    l2_regularization:  float      = 1e-4
    batch_norm:         bool       = True
    activation:         str        = "relu"


@dataclass
class TwoTowerConfig:
    """Full Two-Tower model configuration."""

    # Towers
    user_tower:    TowerConfig = field(default_factory=TowerConfig)
    product_tower: TowerConfig = field(default_factory=TowerConfig)

    # Shared output embedding dimension (must match both towers' last layer)
    output_embedding_dim: int = 64

    # Training
    learning_rate:     float = 1e-3
    batch_size:        int   = 2048
    num_epochs:        int   = 20
    early_stopping_patience: int = 3

    # Loss
    loss:             str   = "batch_softmax"   # or "bpr", "sampled_softmax"
    temperature:      float = 0.07              # softmax temperature

    # Negative sampling
    num_negatives:    int   = 10                # in-batch negatives multiplier
    hard_negative_ratio: float = 0.3           # fraction of hard negatives

    # Categorical features vocab sizes (computed from data at runtime)
    user_vocab_sizes:    Dict[str, int] = field(default_factory=lambda: {
        "user_id":            100_000,
        "gender":             3,
        "country":            200,
        "acquisition_source": 10,
        "top_category_1":     30,
        "price_tier":         4,
    })
    product_vocab_sizes: Dict[str, int] = field(default_factory=lambda: {
        "product_id":    50_000,
        "category":      30,
        "brand":         1_000,
        "department":    3,
        "price_tier":    4,
        "lifecycle_stage": 4,
    })

    # Numerical feature lists
    user_numerical_features: List[str] = field(default_factory=lambda: [
        "age", "account_age_days", "total_orders", "total_spend",
        "avg_order_value", "days_since_last_order",
        "rfm_composite_score", "rfm_recency_score",
        "rfm_frequency_score", "rfm_monetary_score",
        "orders_last_30d", "spend_last_30d", "is_active_30d",
        "return_rate", "view_to_purchase_rate",
        "total_sessions", "product_views",
    ])
    product_numerical_features: List[str] = field(default_factory=lambda: [
        "retail_price", "margin_rate", "total_purchases",
        "unique_buyers", "popularity_score", "return_rate",
        "avg_discount_rate", "sales_last_30d",
        "log_total_purchases", "is_new_product", "is_high_return",
    ])

    # ANN Index retrieval
    num_candidates:    int   = 100             # candidates per user at retrieval
    ann_num_leaves:    int   = 500             # ScaNN / FAISS num_leaves


# ─── LightGBM Ranking Config 

@dataclass
class RankingConfig:
    """LightGBM LambdaRank configuration for the Ranking stage."""

    # Core hyperparameters
    objective:          str   = "lambdarank"
    metric:             str   = "ndcg"
    ndcg_eval_at:       List[int] = field(default_factory=lambda: [5, 10, 20])
    learning_rate:      float = 0.05
    num_leaves:         int   = 127
    max_depth:          int   = -1             # -1 = unlimited
    min_child_samples:  int   = 20
    n_estimators:       int   = 1000
    early_stopping_rounds: int = 50
    subsample:          float = 0.8
    colsample_bytree:   float = 0.8
    reg_alpha:          float = 0.1           # L1
    reg_lambda:         float = 1.0           # L2
    label_gain:         List[float] = field(default_factory=lambda: [0, 1, 3, 7, 15])
    verbosity:          int   = -1

    # Feature columns used for ranking (subset of training_dataset columns)
    feature_columns: List[str] = field(default_factory=lambda: [
        # User features
        "age", "account_age_days", "total_orders", "total_spend",
        "avg_order_value", "days_since_last_order",
        "rfm_composite_score", "rfm_recency_score",
        "rfm_frequency_score", "rfm_monetary_score",
        "orders_last_30d", "spend_last_30d", "is_active_30d",
        "return_rate_u", "view_to_purchase_rate",
        "total_sessions", "product_views",
        # Product features
        "retail_price", "margin_rate", "total_purchases_p",
        "unique_buyers", "popularity_score", "return_rate_p",
        "avg_discount_rate", "sales_last_30d_p",
        "log_total_purchases", "is_new_product", "is_high_return",
        # Interaction features
        "purchase_count", "view_count", "cart_count",
        "total_spend_interaction", "implicit_score_normalised",
    ])
    label_column: str = "label"
    group_column: str = "user_id"              # for LambdaRank grouping


# ─── Evaluation Config 
@dataclass
class EvaluationConfig:
    """Thresholds and settings for model evaluation."""

    # Ranking metrics computed at these K values
    k_values:             List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Minimum acceptable metric values for model promotion
    min_ndcg_at_10:       float = 0.30
    min_precision_at_10:  float = 0.15
    min_recall_at_10:     float = 0.10
    min_map_at_10:        float = 0.12
    min_coverage:         float = 0.20        # fraction of catalogue recommended ≥ once

    # Test set split
    test_size:            float = 0.15
    val_size:             float = 0.10
    time_based_split:     bool  = True        # use temporal split (no future leakage)
    split_date:           Optional[str] = None  # ISO date; None = auto (latest 15%)

    # Diversity / serendipity thresholds
    min_intra_list_diversity: float = 0.40
    min_catalog_coverage:     float = 0.15


# ─── Model Registry Config 
@dataclass
class RegistryConfig:
    """Vertex AI Model Registry and artifact storage settings."""

    project_id:          str  = ""
    region:              str  = "us-central1"
    artifact_bucket:     str  = ""            # GCS bucket (no gs:// prefix)
    artifact_prefix:     str  = "recommendation_system/models"

    # Vertex AI
    vertex_model_name:   str  = "thelook-recommender"
    serving_container:   str  = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"

    # Versioning
    model_version:       str  = "1.0.0"
    champion_alias:      str  = "champion"
    challenger_alias:    str  = "challenger"

    # Promotion gate: model must pass EvaluationConfig thresholds to be promoted
    auto_promote:        bool = True


# ─── Singleton accessors 

DEFAULT_TWO_TOWER_CONFIG = TwoTowerConfig()
DEFAULT_RANKING_CONFIG   = RankingConfig()
DEFAULT_EVAL_CONFIG      = EvaluationConfig()