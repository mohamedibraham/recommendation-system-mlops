"""
Full End-to-End Recommendation System Pipeline
===============================================
Vertex AI / Kubeflow Pipelines (KFP v2) definition.
Scheduling:
  This pipeline is designed to run daily via Cloud Scheduler → Vertex AI.
  Run: python pipeline/full_pipeline.py --compile-only  to produce pipeline.yaml
  Run: python pipeline/full_pipeline.py --submit        to submit to Vertex AI

Environment Variables Required:
  PROJECT_ID          GCP project ID
  REGION              GCP region (default: us-central1)
  ARTIFACT_BUCKET     GCS bucket for model artifacts (no gs://)
  PIPELINE_BUCKET     GCS bucket for KFP pipeline root
  SERVICE_ACCOUNT     Vertex AI service account email
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from kfp import compiler, dsl
from kfp.dsl import (
    Dataset,
    If,
    Input,
    Metrics,
    Model,
    Output,
    pipeline,
)

# ── Import all KFP components 
from src.features.feature_pipeline      import feature_engineering_pipeline
from src.training.model_trainer         import model_training_component
from src.evaluation.model_evaluator     import model_evaluation_component
from src.registration.model_registry    import model_registration_component

logger = logging.getLogger(__name__)

# ── Data validation component (re-use existing component) 
# Inline definition to avoid circular imports; mirrors data_validation.py logic

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "google-cloud-bigquery==3.11.4",
        "pyarrow==13.0.0",
        "kfp==2.4.0",
    ],
)
def data_validation_component(
    project_id:        str,
    max_null_rate:     int,
    metrics:           Output[Metrics],
    validated_dataset: Output[Dataset],
) -> str:
    """
    Validates raw BigQuery data quality before feature engineering.
    Mirrors the logic in src/data_validation.py.
    Returns JSON validation result and passes validated_dataset artifact.
    """
    import json
    import logging
    from google.cloud import bigquery

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    client = bigquery.Client(project=project_id)
    BQ     = "bigquery-public-data.thelook_ecommerce"

    checks  = {}
    errors  = []
    stats   = {}
    passed  = True

    # ── 1. Missing values 
    missing_sql = f"""
    SELECT
        COUNTIF(user_id    IS NULL) AS missing_users,
        COUNTIF(product_id IS NULL) AS missing_products,
        COUNTIF(sale_price IS NULL) AS missing_price
    FROM `{BQ}.order_items`
    """
    row = client.query(missing_sql).to_dataframe().iloc[0]
    total_missing = int(row["missing_users"] + row["missing_products"] + row["missing_price"])
    stats.update({
        "missing_users":    int(row["missing_users"]),
        "missing_products": int(row["missing_products"]),
        "missing_prices":   int(row["missing_price"]),
    })
    checks["missing_values"] = total_missing <= max_null_rate
    if not checks["missing_values"]:
        passed = False
        errors.append(f"Total missing values: {total_missing} > threshold {max_null_rate}")

    # ── 2. Duplicate records 
    dup_sql = f"""
    SELECT COUNT(*) AS dup_count FROM (
        SELECT order_id, product_id, user_id, COUNT(*) AS c
        FROM `{BQ}.order_items`
        GROUP BY 1, 2, 3 HAVING c > 1
    )
    """
    dup_count = int(client.query(dup_sql).to_dataframe().iloc[0]["dup_count"])
    stats["duplicate_records"] = dup_count
    checks["duplicate_records"] = dup_count == 0
    if dup_count > 0:
        passed = False
        errors.append(f"Found {dup_count} duplicate records")

    # ── 3. Negative prices 
    neg_sql = f"""
    SELECT COUNT(*) AS neg_count
    FROM `{BQ}.order_items` WHERE sale_price < 0
    """
    neg_count = int(client.query(neg_sql).to_dataframe().iloc[0]["neg_count"])
    stats["negative_prices"] = neg_count
    checks["negative_prices"] = neg_count == 0
    if neg_count > 0:
        passed = False
        errors.append(f"Found {neg_count} negative prices")

    # ── 4. Broken relationships 
    orphan_sql = f"""
    SELECT COUNT(*) AS orphan_count
    FROM `{BQ}.order_items` oi
    LEFT JOIN `{BQ}.products` p ON oi.product_id = p.id
    WHERE p.id IS NULL
    """
    orphan_count = int(client.query(orphan_sql).to_dataframe().iloc[0]["orphan_count"])
    stats["orphaned_order_items"] = orphan_count
    checks["referential_integrity"] = orphan_count == 0
    if orphan_count > 0:
        errors.append(f"Found {orphan_count} order_items with no matching product")

    # ── Save validated dataset 
    validated_sql = f"""
    SELECT *
    FROM `{BQ}.order_items`
    WHERE user_id IS NOT NULL
      AND product_id IS NOT NULL
      AND sale_price >= 0
    LIMIT 100
    """
    validated_df = client.query(validated_sql).to_dataframe()
    validated_df.to_parquet(validated_dataset.path, index=False)

    # ── Log metrics 
    for key, val in stats.items():
        metrics.log_metric(key, val)
    metrics.log_metric("validation_passed", int(passed))

    result = {
        "passed":  passed,
        "checks":  checks,
        "errors":  errors,
        "stats":   stats,
    }
    log.info("Data validation: %s", "PASSED" if passed else "FAILED")
    return json.dumps(result, indent=2)


# Conditional Deployment Component


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.0",
        "kfp==2.4.0",
    ],
)
def deploy_to_endpoint_component(
    project_id:          str,
    region:              str,
    endpoint_name:       str,
    registration_output: Input[Dataset],
    deployment_metrics:  Output[Metrics],
) -> str:
    """
    Deploy the champion model to a Vertex AI Endpoint.
    Only runs if model_registration_component promoted the model.

    Args:
        project_id:          GCP project.
        region:              Deployment region.
        endpoint_name:       Vertex AI Endpoint display name.
        registration_output: Registration record from model_registration_component.
        deployment_metrics:  KFP Metrics artifact.

    Returns:
        JSON deployment result.
    """
    import json
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Load registration record
    reg_df  = pd.read_parquet(registration_output.path)
    reg_row = reg_df.iloc[0].to_dict()

    status = reg_row.get("status", "")
    if status != "REGISTERED":
        log.info("Model not registered (status=%s). Skipping deployment.", status)
        result = {"status": "SKIPPED", "reason": status}
        deployment_metrics.log_metric("deployed", 0)
        return json.dumps(result)

    vertex_model_name = reg_row.get("vertex_model_name")
    log.info("Deploying model: %s to endpoint: %s", vertex_model_name, endpoint_name)

    try:
        from google.cloud import aiplatform
        aiplatform.init(project=project_id, location=region)

        # Get or create endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"',
            project=project_id, location=region,
        )
        if endpoints:
            endpoint = endpoints[0]
            log.info("Found existing endpoint: %s", endpoint.resource_name)
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=project_id, location=region,
            )
            log.info("Created new endpoint: %s", endpoint.resource_name)

        # Deploy model
        model = aiplatform.Model(model_name=vertex_model_name)
        model.deploy(
            endpoint             = endpoint,
            deployed_model_display_name = f"recommender_{reg_row.get('version_id')}",
            machine_type         = "n1-standard-4",
            min_replica_count    = 1,
            max_replica_count    = 5,
            traffic_percentage   = 100,
        )
        log.info("✅ Model deployed to endpoint: %s", endpoint.resource_name)

        result = {
            "status":               "DEPLOYED",
            "endpoint_name":        endpoint.resource_name,
            "model_version":        reg_row.get("version_id"),
            "traffic_percentage":   100,
        }
        deployment_metrics.log_metric("deployed", 1)
        deployment_metrics.log_metric(
            "ndcg_at_10", float(reg_row.get("ndcg_at_10", 0.0))
        )

    except Exception as e:
        log.error("Deployment failed: %s", e)
        result = {"status": "FAILED", "error": str(e)}
        deployment_metrics.log_metric("deployed", 0)

    return json.dumps(result, indent=2)


# Full Pipeline Definition

@pipeline(
    name        = "thelook-recommendation-system-pipeline",
    description = (
        "End-to-end ML pipeline for TheLook Ecommerce Recommendation System. "
        "Stages: data validation → feature engineering → model training → "
        "evaluation → registration → deployment."
    ),
)
def recommendation_pipeline(
    # ── Infrastructure 
    project_id:          str   = "your-gcp-project",
    region:              str   = "us-central1",
    artifact_bucket:     str   = "your-ml-artifacts-bucket",

    # ── Data 
    feature_store_dataset: str = "recommendation_feature_store",
    max_null_rate:         int = 100,

    # ── Model versioning 
    model_version:         str = "1.0.0",
    feature_version:       str = "v1",

    # ── Serving 
    endpoint_name:         str = "thelook-recommender-endpoint",
    deploy_on_promotion:   bool = True,
) -> None:
    """
    Full end-to-end pipeline for the TheLook Recommendation System.

    Parameters are injectable at runtime via Vertex AI Pipeline parameters,
    allowing different configurations for dev / staging / production runs.
    """

    
    # Stage 1: Data Validation
    # ─────────────────────────────────────────────────────────────────────
    validation_task = data_validation_component(
        project_id    = project_id,
        max_null_rate = max_null_rate,
    )
    validation_task.set_display_name(" Data Validation")
    validation_task.set_retry(num_retries=2, backoff_duration="30s")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2: Feature Engineering & Feature Store Materialisation
    # ─────────────────────────────────────────────────────────────────────
    feature_task = (
        feature_engineering_pipeline(
            project_id       = project_id,
            output_dataset   = feature_store_dataset,
            feature_version  = feature_version,
        )
        .after(validation_task)
        .set_display_name("  Feature Engineering")
        .set_cpu_limit("4")
        .set_memory_limit("16G")
        .set_retry(num_retries=1, backoff_duration="60s")
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 3: Model Training (Two-Tower + LightGBM Ranking)
    # ─────────────────────────────────────────────────────────────────────
    training_task = (
        model_training_component(
            project_id       = project_id,
            training_dataset = feature_task.outputs["training_dataset"],
        )
        .after(feature_task)
        .set_display_name("🏋️  Model Training")
        .set_cpu_limit("8")
        .set_memory_limit("32G")
        .set_retry(num_retries=1, backoff_duration="120s")
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 4: Model Evaluation
    # ─────────────────────────────────────────────────────────────────────
    evaluation_task = (
        model_evaluation_component(
            test_split        = training_task.outputs["test_split"],
            ranking_model_dir = training_task.outputs["ranking_model_dir"],
            model_version     = model_version,
        )
        .after(training_task)
        .set_display_name(" Model Evaluation")
        .set_cpu_limit("4")
        .set_memory_limit("16G")
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5: Model Registration
    # ─────────────────────────────────────────────────────────────────────
    registration_task = (
        model_registration_component(
            project_id        = project_id,
            region            = region,
            artifact_bucket   = artifact_bucket,
            model_version     = model_version,
            ranking_model_dir = training_task.outputs["ranking_model_dir"],
            evaluation_report = evaluation_task.outputs["evaluation_report"],
            feature_version   = feature_version,
        )
        .after(evaluation_task)
        .set_display_name(" Model Registration")
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 6: Conditional Deployment
    # Only runs if the model was promoted (evaluation passed + champion)
    # ─────────────────────────────────────────────────────────────────────
    with dsl.If(
        deploy_on_promotion == True,
        name="Deploy if promotion enabled",
    ):
        deploy_task = (
            deploy_to_endpoint_component(
                project_id          = project_id,
                region              = region,
                endpoint_name       = endpoint_name,
                registration_output = registration_task.outputs["registration_output"],
            )
            .after(registration_task)
            .set_display_name("🚀 Deploy to Endpoint")
        )


# ════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ════════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """
    Compiles and submits the recommendation pipeline to Vertex AI.

    Usage:
        runner = PipelineRunner(
            project_id     = "my-project",
            region         = "us-central1",
            pipeline_bucket= "my-pipeline-bucket",
            artifact_bucket= "my-artifacts-bucket",
            service_account= "vertex-sa@my-project.iam.gserviceaccount.com",
        )
        runner.compile()
        runner.submit(model_version="1.0.1", enable_caching=True)
    """

    COMPILED_PATH = "recommendation_pipeline.yaml"

    def __init__(
        self,
        project_id:       str,
        region:           str,
        pipeline_bucket:  str,
        artifact_bucket:  str,
        service_account:  Optional[str] = None,
    ) -> None:
        self.project_id       = project_id
        self.region           = region
        self.pipeline_root    = f"gs://{pipeline_bucket}/pipelines"
        self.artifact_bucket  = artifact_bucket
        self.service_account  = service_account

    def compile(self, output_path: str = COMPILED_PATH) -> str:
        """
        Compile the pipeline to a YAML artifact.

        Args:
            output_path: Where to write the compiled YAML.

        Returns:
            Path to the compiled YAML file.
        """
        logger.info("Compiling pipeline → %s …", output_path)
        compiler.Compiler().compile(
            pipeline_func = recommendation_pipeline,
            package_path  = output_path,
        )
        logger.info("Pipeline compiled successfully → %s", output_path)
        return output_path

    def submit(
        self,
        model_version:    str  = "1.0.0",
        feature_version:  str  = "v1",
        enable_caching:   bool = True,
        sync:             bool = False,
    ) -> str:
        """
        Submit the compiled pipeline to Vertex AI Pipelines.

        Args:
            model_version:   Semantic version for this run.
            feature_version: Feature store version to use.
            enable_caching:  Reuse cached component outputs when inputs unchanged.
            sync:            Wait for pipeline completion if True.

        Returns:
            Vertex AI pipeline run resource name.
        """
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
        except ImportError:
            raise RuntimeError(
                "google-cloud-aiplatform is required to submit pipelines. "
                "Install with: pip install google-cloud-aiplatform"
            )

        # Auto-compile if YAML not found
        if not os.path.exists(self.COMPILED_PATH):
            self.compile()

        ts         = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        job_id     = f"thelook-recommender-{model_version}-{ts}"

        logger.info("Submitting pipeline job: %s …", job_id)
        job = aiplatform.PipelineJob(
            display_name        = job_id,
            template_path       = self.COMPILED_PATH,
            pipeline_root       = self.pipeline_root,
            job_id              = job_id,
            enable_caching      = enable_caching,
            parameter_values    = {
                "project_id":       self.project_id,
                "region":           self.region,
                "artifact_bucket":  self.artifact_bucket,
                "model_version":    model_version,
                "feature_version":  feature_version,
            },
        )
        job.submit(service_account=self.service_account)

        if sync:
            job.wait()
            logger.info("Pipeline job completed: %s", job.state)
        else:
            logger.info("Pipeline job submitted: %s", job.resource_name)

        return job.resource_name


# ════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="TheLook Recommendation System Pipeline"
    )
    parser.add_argument(
        "--compile-only", action="store_true",
        help="Only compile to YAML, do not submit.",
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="Compile and submit to Vertex AI.",
    )
    parser.add_argument("--project-id",       default=os.getenv("PROJECT_ID", ""))
    parser.add_argument("--region",           default=os.getenv("REGION", "us-central1"))
    parser.add_argument("--pipeline-bucket",  default=os.getenv("PIPELINE_BUCKET", ""))
    parser.add_argument("--artifact-bucket",  default=os.getenv("ARTIFACT_BUCKET", ""))
    parser.add_argument("--service-account",  default=os.getenv("SERVICE_ACCOUNT", ""))
    parser.add_argument("--model-version",    default="1.0.0")
    parser.add_argument("--feature-version",  default="v1")
    parser.add_argument("--output-yaml",      default="recommendation_pipeline.yaml")

    args = parser.parse_args()

    runner = PipelineRunner(
        project_id      = args.project_id,
        region          = args.region,
        pipeline_bucket = args.pipeline_bucket,
        artifact_bucket = args.artifact_bucket,
        service_account = args.service_account,
    )

    if args.compile_only or args.submit:
        yaml_path = runner.compile(output_path=args.output_yaml)
        print(f" Compiled → {yaml_path}")

    if args.submit:
        resource_name = runner.submit(
            model_version   = args.model_version,
            feature_version = args.feature_version,
        )
        print(f" Submitted → {resource_name}")
    elif not args.compile_only and not args.submit:
        parser.print_help()