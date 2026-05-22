from kfp import dsl
from kfp.dsl import Input, Output, Model, Metrics, HTML
import os
from dotenv import load_dotenv


load_dotenv()

GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")

CUSTOM_BASE_IMAGE = f"{os.environ.get('REGION')}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"

@dsl.component(
    base_image=CUSTOM_BASE_IMAGE
)
def evaluate_xgbranker(
    project_id: str,
    bq_test_table_uri: str,
    trained_model: Input[Model],
    metrics: Output[Metrics],
    evaluation_report: Output[HTML],
    ndcg_5_threshold_override: float = 0.0, 
) -> str:
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import ndcg_score
    import logging
    from src.utils.bq_utils import BigQueryManager
    from config.model_config import default_model_config as cfg
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    threshold = ndcg_5_threshold_override if ndcg_5_threshold_override > 0.0 else cfg.evaluation.ndcg_5_threshold
    
    table_id = bq_test_table_uri.replace("bq://", "")
    bq_client = BigQueryManager(project_id=project_id)
    
    query = f"SELECT * FROM `{table_id}` ORDER BY {cfg.features.group_column}"
    df = bq_client.Query_To_Dataframe(query=query)

    for col in cfg.features.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype('category')

    feature_cols = [c for c in df.columns if c not in cfg.features.ignore_columns]
    X_test, y_test = df[feature_cols], df[cfg.features.target_column]

    ranker = xgb.XGBRanker()
    ranker.load_model(trained_model.path)

    df['predictions'] = ranker.predict(X_test)
    
    ndcg_scores = []
    total_sessions = 0
    sessions_with_positive = 0
    
    for _, group in df.groupby(cfg.features.group_column):
        total_sessions += 1
        if group[cfg.features.target_column].sum() > 0:
            sessions_with_positive += 1
            if len(group) > 1: 
                true_rel = [group[cfg.features.target_column].values]
                preds = [group['predictions'].values]
                score = ndcg_score(true_rel, preds, k=5)
                ndcg_scores.append(score)

    mean_ndcg_5 = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    metrics.log_metric("test_mean_ndcg_5", float(mean_ndcg_5))
    metrics.log_metric("target_threshold", float(threshold))

    if mean_ndcg_5 >= threshold:
        deploy_decision = "approve"
        logger.info(f"Model Approved! NDCG@5: {mean_ndcg_5:.4f} >= Threshold: {threshold}")
    else:
        deploy_decision = "reject"
        logger.warning(f"Model Rejected. NDCG@5: {mean_ndcg_5:.4f} < Threshold: {threshold}")
    
    
    html_content = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 600px; }}
                h2 {{ color: #202124; }}
                .status {{ font-weight: bold; padding: 8px 12px; border-radius: 4px; display: inline-block; }}
                .approve {{ background-color: #e6f4ea; color: #137333; }}
                .reject {{ background-color: #fce8e6; color: #c5221f; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f1f3f4; }}
            </style>
        </head>
        <body>
            <div class="card">
                <h2>XGBRanker Evaluation Report</h2>
                <p>Status: <span class="status {deploy_decision}">{deploy_decision.upper()}</span></p>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Mean NDCG @ 5</td><td>{mean_ndcg_5:.4f}</td></tr>
                    <tr><td>Target Threshold</td><td>{threshold:.4f}</td></tr>
                    <tr><td>Total Sessions Processed</td><td>{total_sessions}</td></tr>
                    <tr><td>Sessions with Actions</td><td>{sessions_with_positive}</td></tr>
                </table>
            </div>
        </body>
    </html>
    """
    
    with open(evaluation_report.path, "w") as f:
        f.write(html_content)

    return deploy_decision