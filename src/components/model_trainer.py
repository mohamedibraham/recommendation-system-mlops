from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics
import os
from dotenv import load_dotenv


load_dotenv()

GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")

CUSTOM_BASE_IMAGE = f"{os.environ.get('REGION')}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"

@dsl.component(
    base_image=CUSTOM_BASE_IMAGE
)
def train_xgbranker(
    project_id: str,
    bq_table_uri: str, 
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 0, 
    learning_rate: float = 0.0,
    max_depth: int = 0,
):
    import pandas as pd
    import xgboost as xgb
    import logging
    

    from src.utils.bq_utils import BigQueryManager
    from config.model_config import default_model_config as cfg

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    
    if n_estimators > 0: cfg.hyperparams.n_estimators = n_estimators
    if learning_rate > 0: cfg.hyperparams.learning_rate = learning_rate
    if max_depth > 0: cfg.hyperparams.max_depth = max_depth

    
    table_id = bq_table_uri.replace("bq://", "")
    bq_client = BigQueryManager(project_id=project_id)
    
    query = f"SELECT * FROM `{table_id}` ORDER BY session_start_time, {cfg.features.group_column}"
    logger.info(f"Loading data from BigQuery: {table_id}")
    df = bq_client.Query_To_Dataframe(query=query)

    
    for col in cfg.features.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype('category')

    feature_cols = [c for c in df.columns if c not in cfg.features.ignore_columns and c != 'session_start_time']

    
    unique_groups = df[cfg.features.group_column].unique()
    split_idx = int(len(unique_groups) * (1 - cfg.evaluation.test_size))
    
    train_groups = set(unique_groups[:split_idx])
    train_mask = df[cfg.features.group_column].isin(train_groups)
    
    train_df = df[train_mask].sort_values(cfg.features.group_column)
    val_df = df[~train_mask].sort_values(cfg.features.group_column)

    X_train, y_train = train_df[feature_cols], train_df[cfg.features.target_column]
    X_val, y_val = val_df[feature_cols], val_df[cfg.features.target_column]

    group_train = train_df.groupby(cfg.features.group_column).size().values
    group_val = val_df.groupby(cfg.features.group_column).size().values

    
    logger.info("Initializing XGBRanker with config parameters...")
    ranker = xgb.XGBRanker(
        tree_method=cfg.hyperparams.tree_method,
        enable_categorical=True,
        objective=cfg.hyperparams.objective,
        n_estimators=cfg.hyperparams.n_estimators,
        learning_rate=cfg.hyperparams.learning_rate,
        max_depth=cfg.hyperparams.max_depth,
        early_stopping_rounds=cfg.hyperparams.early_stopping_rounds,
        eval_metric=cfg.hyperparams.eval_metric
    )

    logger.info("Starting model training...")
    ranker.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[group_train, group_val],
        verbose=False 
    )

    best_iteration = ranker.best_iteration
    best_ndcg_5 = ranker.evals_result()['validation_1']['ndcg@5'][best_iteration]
    
    metrics.log_metric("val_ndcg_5", float(best_ndcg_5))
    model.metadata["framework"] = "XGBoost"
    ranker.save_model(model.path)