from kfp import dsl
from kfp.dsl import Input, Dataset
import os
from dotenv import load_dotenv


load_dotenv()

GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")

CUSTOM_BASE_IMAGE = f"{os.environ.get('REGION')}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"

@dsl.component(
    base_image=CUSTOM_BASE_IMAGE
)
def feature_registration(
    project_id: str,
    region: str,
    bq_dataset_name: str,
    feature_group_name: str,
    extracted_features: Input[Dataset],
) -> str:
    
    import pandas as pd
    from src.utils.bq_utils import BigQueryManager
    from google.cloud import aiplatform
    from google.api_core.exceptions import AlreadyExists
    import logging
    

    logger = logging.getLogger(__name__)

    logger.info("Loading extracted features...")
    df = pd.read_parquet(extracted_features.path)

    
    table_id = f"{project_id}.{bq_dataset_name}.{feature_group_name}"

    bq_client = BigQueryManager(project=project_id)
    
    logger.info(f"Uploading features to BigQuery Table: {table_id}")

    bq_client = BigQueryManager(project_id=project_id)
    bq_client.dataframe_to_table(
    df=df,
    destination_table=table_id,
    write_disposition="WRITE_TRUNCATE"
)

    aiplatform.init(project=project_id, location=region)

    

    entity_cols = ["user_id", "product_id"]
    
    from google.api_core.exceptions import NotFound
    try:
         feature_group = aiplatform.FeatureGroup(feature_group_name).gca_resource
         
    except NotFound:
       
        logger.info(f"Creating new Feature Group: '{feature_group_name}'")
        feature_group = aiplatform.FeatureGroup.create(
            name=feature_group_name,
            bigquery_source=f"bq://{table_id}",
            entity_id_columns=entity_cols,
        )

   
    ignore_columns = ["qid", "user_id", "product_id", "target_label"]
    feature_names = [col for col in df.columns if col not in ignore_columns]

    logger.info(f"Registering {len(feature_names)} features in Vertex AI...")
    for feature in feature_names:
        try:
            feature_group.create_feature(
                name=feature,
                description=f"Automated feature extraction for {feature}"
            )
            logger.info(f" - Registered: {feature}")
        except AlreadyExists:
            logger.info(f" - Feature {feature} already exists, skipping.")

    return f"bq://{table_id}"