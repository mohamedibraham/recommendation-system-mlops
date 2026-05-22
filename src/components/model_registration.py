from kfp import dsl
from kfp.dsl import Input, Model
import os
from dotenv import load_dotenv


load_dotenv()

GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")

CUSTOM_BASE_IMAGE = f"{os.environ.get('REGION')}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"

@dsl.component(
    base_image=CUSTOM_BASE_IMAGE,
)
def register_model(
    project_id: str,
    region: str,
    trained_model: Input[Model],
    model_display_name_override: str = "",
) -> str:
    from google.cloud import aiplatform
    import logging
    
    
    from config.model_config import default_model_config as cfg

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    
    display_name = model_display_name_override if model_display_name_override else cfg.deployment.model_display_name
    container_uri = cfg.deployment.serving_container_image

    aiplatform.init(project=project_id, location=region)

    
    model_artifact_dir = trained_model.uri.rsplit('/', 1)[0]

    logger.info(f"Uploading model: {display_name} to Vertex AI Model Registry")
    logger.info(f"Using serving container: {container_uri}")

    
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_artifact_dir,
        serving_container_image_uri=container_uri,
        description="XGBRanker model for product recommendation powered by MLOps Pipeline"
    )

    logger.info(f"Model uploaded successfully. Model Resource Name: {model.resource_name}")
    return model.resource_name