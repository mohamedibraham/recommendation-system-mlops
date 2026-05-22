# config/model_config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class FeaturesConfig:
    categorical_columns: List[str] = field(default_factory=lambda: [
        'user_gender', 'user_country', 'brand'
    ])
    ignore_columns: List[str] = field(default_factory=lambda: [
        'qid', 'user_id', 'product_id', 'target_label'
    ])
    target_column: str = 'target_label'
    group_column: str = 'qid'

@dataclass
class XGBRankerHyperparams:
    n_estimators: int = 150
    learning_rate: float = 0.1
    max_depth: int = 5
    tree_method: str = "hist"
    objective: str = "rank:ndcg"
    early_stopping_rounds: int = 15
    eval_metric: List[str] = field(default_factory=lambda: ["ndcg@5", "ndcg@10"])

@dataclass
class EvaluationConfig:
    test_size: float = 0.2
    random_state: int = 42
    ndcg_5_threshold: float = 0.65  

@dataclass
class DeploymentConfig:
    model_display_name: str = "xgbranker-product-recommendation"
    serving_container_image: str = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"

@dataclass
class ModelConfig:
    
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    hyperparams: XGBRankerHyperparams = field(default_factory=XGBRankerHyperparams)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

default_model_config = ModelConfig()