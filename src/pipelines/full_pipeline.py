import os
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform

from src.components.data_validation import data_validation
from src.features.Feature_extraction import feature_extraction
from src.feature_store.feature_registry import feature_registration
from src.components.model_trainer import train_xgbranker
from src.components.model_evaluation import evaluate_xgbranker
from src.components.model_registration import register_model

@dsl.pipeline(
    name="xgbranker-e2e-production-pipeline",
    description="An end-to-end production MLOps pipeline for product ranking using XGBRanker, BigQuery, and Vertex AI Feature Store.",
)
def xgbranker_pipeline(
    project_id: str,
    region: str,
    bq_dataset_name: str,
    feature_group_name: str,
    max_null_rate: int = 100,
    n_estimators: int = 150,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    ndcg_5_threshold: float = 0.65,
    model_display_name: str = "xgbranker-product-recommendation",
):
    
    validation_task = data_validation(
        project_id=project_id,
        max_null_rate=max_null_rate
    )

    
    extraction_task = feature_extraction(
        project_id=project_id,
        validated_dataset=validation_task.outputs["validated_dataset"]
    )

    
    registration_task = feature_registration(
        project_id=project_id,
        region=region,
        bq_dataset_name=bq_dataset_name,
        feature_group_name=feature_group_name,
        extracted_features=extraction_task.outputs["extracted_features"]
    )

    
    training_task = train_xgbranker(
        project_id=project_id,
        bq_table_uri=registration_task.output,  
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )

    
    evaluation_task = evaluate_xgbranker(
        project_id=project_id,
        bq_test_table_uri=registration_task.output,  
        trained_model=training_task.outputs["model"],
        ndcg_5_threshold_override=ndcg_5_threshold
    )

    
    with dsl.Condition(
        evaluation_task.outputs["Output"] == "approve", 
        name="Performance-Gate-Approval"
    ):
        
        
        register_task = register_model(
            project_id=project_id,
            region=region,
            model_display_name_override=model_display_name,
            trained_model=training_task.outputs["model"]
        )
        
        
if __name__ == "__main__":
   
    PIPELINE_PACKAGE_PATH = "xgbranker_production_pipeline.yaml"
    
    print(f"Compiling pipeline into {PIPELINE_PACKAGE_PATH}...")
    compiler.Compiler().compile(
        pipeline_func=xgbranker_pipeline,
        package_path=PIPELINE_PACKAGE_PATH
    )
    print("Pipeline compiled successfully.")

    