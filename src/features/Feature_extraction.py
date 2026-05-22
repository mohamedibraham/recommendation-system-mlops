from kfp import dsl
from kfp.dsl import Dataset , Metrics , Output , Input
import os
from dotenv import load_dotenv


load_dotenv()

GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")

CUSTOM_BASE_IMAGE = f"{os.environ.get('REGION')}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"

@dsl.component(
    base_image=CUSTOM_BASE_IMAGE
)

def feature_extraction(
    project_id : str ,
    validated_dataset: Input[Dataset], 
    extracted_features: Output[Dataset],
    metrics: Output[Metrics],
):
    
    
    from src.utils.bq_utils import BigQueryManager
    from src.queries.feature_queries import Feature_Queries
    from config.model_config import default_model_config as cfg
    import logging

    logger = logging.getLogger(__name__)

    client = BigQueryManager(project_id=project_id)
    
    logger.info("Assembling the full feature extraction query...")

    full_query = Feature_Queries.get_full_query()
    
    logger.info("Executing query on BigQuery. This might take a few minutes...")
    
    features_df = client.Query_To_Dataframe(query=full_query)
    
    
    for col in cfg.features.categorical_columns:
        if col in features_df.columns:
            
            features_df[col] = features_df[col].fillna("Unknown")
            
            features_df[col] = features_df[col].astype('category')
            
    print(f"Extraction complete. Shape of features dataframe: {features_df.shape}")
    
    
    features_df.to_parquet(extracted_features.path, index=False)
    
    
    metrics.log_metric("total_training_samples", len(features_df))
    metrics.log_metric("total_features", len(features_df.columns) - 4) 

    positive_count = int(features_df['target_label'].sum())
    metrics.log_metric("positive_samples", positive_count)
    metrics.log_metric("negative_samples", len(features_df) - positive_count)
