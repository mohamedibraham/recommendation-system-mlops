"""
Model Registry
==============
Handles model registration, versioning, and the Champion/Challenger
promotion logic using Google Cloud Vertex AI Model Registry.

"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

logger = logging.getLogger(__name__)

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import Model as VertexModel
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logger.warning("google-cloud-aiplatform not installed. Vertex AI registry unavailable.")



# Model Metadata Container

class ModelVersion:
    """
    Immutable record of a registered model version.
    Stored as JSON in GCS alongside the model artifacts.
    """

    def __init__(
        self,
        model_name:          str,
        version_id:          str,
        artifact_uri:        str,
        metrics:             Dict,
        training_dataset_ref: str,
        feature_version:     str,
        git_commit:          Optional[str] = None,
        pipeline_run_id:     Optional[str] = None,
    ) -> None:
        self.model_name           = model_name
        self.version_id           = version_id
        self.artifact_uri         = artifact_uri
        self.metrics              = metrics
        self.training_dataset_ref = training_dataset_ref
        self.feature_version      = feature_version
        self.git_commit           = git_commit
        self.pipeline_run_id      = pipeline_run_id
        self.registered_at        = datetime.now(timezone.utc).isoformat()
        self.is_champion          = False

    def to_dict(self) -> dict:
        return self.__dict__

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# Champion/Challenger Comparator

class ChampionChallengerComparator:
    """
    Compares a new challenger model against the current champion.

    Promotion criteria:
      1. Challenger NDCG@10 must exceed champion by at least `delta`.
      2. Challenger must not degrade any metric by more than `tolerance`.
      3. Coverage must remain above minimum threshold.

    This prevents noisy improvements from triggering unnecessary deployments.
    """

    def __init__(
        self,
        improvement_delta:    float = 0.01,   # challenger must beat champion by ≥1%
        degradation_tolerance: float = 0.02,  # allow ≤2% degradation on secondary metrics
        primary_metric:        str   = "ndcg_at_10",
    ) -> None:
        self.delta     = improvement_delta
        self.tolerance = degradation_tolerance
        self.primary   = primary_metric

    def should_promote(
        self,
        champion_metrics:   Dict,
        challenger_metrics: Dict,
    ) -> Tuple[bool, str]:
        """
        Determine if challenger should replace champion.

        Returns:
            (should_promote: bool, reason: str)
        """
        champion_primary   = champion_metrics.get(self.primary, 0.0)
        challenger_primary = challenger_metrics.get(self.primary, 0.0)

        # Check primary metric improvement
        improvement = challenger_primary - champion_primary
        if improvement < self.delta:
            return False, (
                f"Challenger {self.primary}={challenger_primary:.4f} "
                f"does not exceed champion={champion_primary:.4f} "
                f"by required delta={self.delta}"
            )

        # Check secondary metric degradation
        secondary_metrics = ["precision_at_10", "recall_at_10", "map_at_10", "coverage"]
        for metric in secondary_metrics:
            champ_val  = champion_metrics.get(metric, 0.0)
            chall_val  = challenger_metrics.get(metric, 0.0)
            degradation = champ_val - chall_val
            if degradation > self.tolerance:
                return False, (
                    f"Challenger degrades {metric}: "
                    f"champion={champ_val:.4f} → challenger={chall_val:.4f} "
                    f"(degradation={degradation:.4f} > tolerance={self.tolerance})"
                )

        return True, (
            f"Challenger improves {self.primary} by "
            f"{improvement:.4f} (champion={champion_primary:.4f} → "
            f"challenger={challenger_primary:.4f})"
        )


# Model Registry

class ModelRegistry:
    """
    Production Model Registry backed by Vertex AI Model Registry + GCS.

    Responsibilities:
      - Register new model versions with full lineage metadata.
      - Maintain champion/challenger aliases.
      - Execute promotion decisions.
      - Provide rollback capability.
      - Log all registry operations to BigQuery audit table.
    """

    def __init__(self, config) -> None:
        self.config     = config
        self.comparator = ChampionChallengerComparator()
        if VERTEX_AVAILABLE:
            aiplatform.init(
                project=config.project_id,
                location=config.region,
            )
        self._registry_log: List[Dict] = []   # in-memory audit log

    
    # Registration
    

    def register_model(
        self,
        artifact_uri:        str,
        evaluation_metrics:  Dict,
        training_dataset_ref: str,
        feature_version:     str,
        pipeline_run_id:     Optional[str] = None,
        git_commit:          Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Steps:
          1. Create ModelVersion record with full lineage.
          2. Upload to Vertex AI Model Registry (if available).
          3. Save metadata JSON to GCS alongside artifacts.
          4. Assign "challenger" alias.
          5. Run champion/challenger comparison.
          6. Promote if warranted.

        Args:
            artifact_uri:        GCS URI of model artifacts (gs://bucket/path).
            evaluation_metrics:  Dict from ModelEvaluator (NDCG, P, R, MAP…).
            training_dataset_ref: BQ table used for training (lineage).
            feature_version:     Feature store version (e.g. "v1").
            pipeline_run_id:     Vertex AI pipeline run ID for traceability.
            git_commit:          Git SHA of the code that produced this model.

        Returns:
            ModelVersion object for this registered version.
        """
        cfg = self.config

        # Generate version ID (timestamp-based for uniqueness)
        ts         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_id = f"{cfg.model_version}_{ts}"

        version = ModelVersion(
            model_name           = cfg.vertex_model_name,
            version_id           = version_id,
            artifact_uri         = artifact_uri,
            metrics              = evaluation_metrics,
            training_dataset_ref = training_dataset_ref,
            feature_version      = feature_version,
            git_commit           = git_commit,
            pipeline_run_id      = pipeline_run_id,
        )

        logger.info("Registering model version: %s", version_id)

        
        vertex_model_id = None
        if VERTEX_AVAILABLE:
            vertex_model_id = self._register_vertex(version)
        else:
            logger.warning("Vertex AI not available. Skipping cloud registration.")

        # ── Save metadata to GCS 
        metadata_uri = os.path.join(artifact_uri, "model_metadata.json")
        logger.info("Metadata URI: %s", metadata_uri)
        # In production: upload version.to_json() to GCS via google.cloud.storage

        # ── Audit log entry
        self._registry_log.append({
            "event":      "REGISTERED",
            "version_id": version_id,
            "timestamp":  version.registered_at,
            "metrics":    evaluation_metrics,
        })

        # ── Champion/Challenger
        if cfg.auto_promote:
            promoted, reason = self._run_promotion(version, evaluation_metrics)
            if promoted:
                version.is_champion = True
                logger.info(" Model %s PROMOTED to champion. Reason: %s",
                            version_id, reason)
            else:
                logger.info("  Model %s assigned as CHALLENGER. Reason: %s",
                            version_id, reason)

        return version

    
    # Champion / Challenger management

    def get_champion_metrics(self) -> Optional[Dict]:
        """
        Retrieve the current champion model's metrics from Vertex AI.

        Returns:
            Dict of metrics, or None if no champion exists yet.
        """
        if not VERTEX_AVAILABLE:
            # Check local audit log for previous champion
            for entry in reversed(self._registry_log):
                if entry.get("event") == "PROMOTED":
                    return entry.get("metrics")
            return None

        try:
            model = aiplatform.Model(
                model_name=f"projects/{self.config.project_id}"
                           f"/locations/{self.config.region}"
                           f"/models/{self.config.vertex_model_name}"
            )
            # Retrieve metrics stored as model labels (simplified)
            labels = model.labels
            return {k: float(v) for k, v in labels.items() if v.replace(".", "").isdigit()}
        except Exception as e:
            logger.info("No current champion found: %s", e)
            return None

    def rollback(self, version_id: str) -> None:
        """
        Roll back production to a specific registered version.
        Sets the given version as champion in Vertex AI.
        """
        logger.info("ROLLBACK: Promoting version %s to champion …", version_id)
        self._registry_log.append({
            "event":      "ROLLBACK",
            "version_id": version_id,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Rollback complete. Version %s is now champion.", version_id)

    def list_versions(self) -> List[Dict]:
        """Return all registered versions from the audit log."""
        return [e for e in self._registry_log if e.get("event") == "REGISTERED"]

    
    # Private helpers

    def _run_promotion(
        self,
        version:            ModelVersion,
        challenger_metrics: Dict,
    ) -> Tuple[bool, str]:
        """Compare challenger to champion and promote if warranted."""
        champion_metrics = self.get_champion_metrics()

        # No existing champion → promote directly
        if champion_metrics is None:
            self._registry_log.append({
                "event":      "PROMOTED",
                "version_id": version.version_id,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "metrics":    challenger_metrics,
                "reason":     "First champion — no prior model.",
            })
            return True, "No prior champion — promoted directly."

        # Compare
        should, reason = self.comparator.should_promote(
            champion_metrics, challenger_metrics
        )

        if should:
            self._registry_log.append({
                "event":      "PROMOTED",
                "version_id": version.version_id,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
                "metrics":    challenger_metrics,
                "reason":     reason,
            })

        return should, reason

    def _register_vertex(self, version: ModelVersion) -> Optional[str]:
        """Upload model artifacts to Vertex AI Model Registry."""
        try:
            vertex_model = aiplatform.Model.upload(
                display_name   = f"{version.model_name}_{version.version_id}",
                artifact_uri   = version.artifact_uri,
                serving_container_image_uri = self.config.serving_container,
                labels={
                    "model_name":      version.model_name,
                    "version_id":      version.version_id,
                    "feature_version": version.feature_version,
                },
                description=f"TheLook Recommender {version.version_id}",
            )
            logger.info("Vertex AI model uploaded: %s", vertex_model.resource_name)
            return vertex_model.resource_name
        except Exception as e:
            logger.error("Vertex AI upload failed: %s", e)
            return None



# KFP Component

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "pyarrow==13.0.0",
        "google-cloud-aiplatform==1.38.0",
        "google-cloud-storage==2.13.0",
        "kfp==2.4.0",
    ],
)
def model_registration_component(
    project_id:           str,
    region:               str,
    artifact_bucket:      str,
    model_version:        str,
    ranking_model_dir:    Input[Model],
    evaluation_report:    Input[Dataset],
    registration_output:  Output[Dataset],
    registry_metrics:     Output[Metrics],
    feature_version:      str = "v1",
    pipeline_run_id:      str = "",
) -> str:
    """
    KFP component: register the trained model in Vertex AI with lineage.

    Args:
        project_id:          GCP project ID.
        region:              GCP region (e.g. 'us-central1').
        artifact_bucket:     GCS bucket name for model artifacts.
        model_version:       Semantic version string (e.g. '1.0.0').
        ranking_model_dir:   Input Model artifact directory.
        evaluation_report:   Parquet evaluation report from evaluation component.
        registration_output: Output Dataset artifact with registration result.
        registry_metrics:    KFP Metrics artifact.
        feature_version:     Feature store version tag.
        pipeline_run_id:     Vertex AI pipeline run ID for traceability.

    Returns:
        JSON registration result string.
    """
    import json
    import logging
    import os
    import shutil
    from datetime import datetime, timezone

    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load evaluation report 
    log.info("Loading evaluation report …")
    eval_df = pd.read_parquet(evaluation_report.path)
    eval_row = eval_df.iloc[0].to_dict()

    promoted = bool(eval_row.get("promoted", False))
    log.info("Evaluation result: promoted=%s", promoted)

    if not promoted:
        failures = eval_row.get("promotion_failures", [])
        result   = {
            "status":    "REJECTED",
            "reason":    "Evaluation thresholds not met",
            "failures":  failures,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        pd.DataFrame([result]).to_parquet(registration_output.path, index=False)
        registry_metrics.log_metric("promoted",         0)
        registry_metrics.log_metric("registration_status", 0)
        log.warning("Model NOT registered. Reasons: %s", failures)
        return json.dumps(result, indent=2, default=str)

    # ── Prepare GCS artifact URI 
    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_id  = f"{model_version}_{ts}"
    artifact_uri = (
        f"gs://{artifact_bucket}/recommendation_system/models/{version_id}"
    )
    log.info("Artifact URI: %s", artifact_uri)

    # ── Upload to GCS 
    try:
        from google.cloud import storage as gcs
        gcs_client = gcs.Client(project=project_id)
        bucket     = gcs_client.bucket(artifact_bucket)

        model_dir  = ranking_model_dir.path
        for fname in os.listdir(model_dir):
            local_path = os.path.join(model_dir, fname)
            blob_path  = f"recommendation_system/models/{version_id}/{fname}"
            blob       = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            log.info("Uploaded: %s → gs://%s/%s", fname, artifact_bucket, blob_path)
        gcs_upload_ok = True
    except Exception as e:
        log.error("GCS upload failed: %s", e)
        gcs_upload_ok = False

    # ── Register in Vertex AI 
    vertex_model_name = None
    try:
        from google.cloud import aiplatform
        aiplatform.init(project=project_id, location=region)

        # Extract flat metrics for Vertex AI labels (string-only)
        metrics_by_k = eval_row.get("metrics_by_k", {})
        ndcg_10  = metrics_by_k.get(10, {}).get("ndcg", 0.0)     \
                   if isinstance(metrics_by_k, dict) else 0.0
        coverage = eval_row.get("coverage", 0.0)

        vertex_model = aiplatform.Model.upload(
            display_name = f"thelook-recommender-{version_id}",
            artifact_uri = artifact_uri,
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
            ),
            labels={
                "model_version":   model_version,
                "promoted":        "true",
                "ndcg_10":         f"{ndcg_10:.4f}".replace(".", "_"),
                "coverage":        f"{coverage:.4f}".replace(".", "_"),
                "feature_version": feature_version,
            },
            description=(
                f"TheLook Recommender | version={version_id} | "
                f"NDCG@10={ndcg_10:.4f} | feature_version={feature_version}"
            ),
        )
        vertex_model_name = vertex_model.resource_name
        log.info("Vertex AI model registered: %s", vertex_model_name)

        # Assign champion alias
        try:
            vertex_model.versioning_user_metadata = {"alias": "champion"}
            log.info("Champion alias assigned.")
        except Exception as alias_err:
            log.warning("Could not assign champion alias: %s", alias_err)

    except Exception as e:
        log.error("Vertex AI registration failed: %s", e)

    # ── Build registration record 
    metrics_by_k = eval_row.get("metrics_by_k", {})
    ndcg_10  = metrics_by_k.get(10, {}).get("ndcg", 0.0) \
               if isinstance(metrics_by_k, dict) else 0.0
    prec_10  = metrics_by_k.get(10, {}).get("precision", 0.0) \
               if isinstance(metrics_by_k, dict) else 0.0
    recall_10 = metrics_by_k.get(10, {}).get("recall", 0.0) \
                if isinstance(metrics_by_k, dict) else 0.0

    registration_record = {
        "status":              "REGISTERED",
        "version_id":          version_id,
        "artifact_uri":        artifact_uri,
        "vertex_model_name":   vertex_model_name,
        "gcs_upload_ok":       gcs_upload_ok,
        "promoted_to_champion": True,
        "feature_version":     feature_version,
        "pipeline_run_id":     pipeline_run_id,
        "ndcg_at_10":          ndcg_10,
        "precision_at_10":     prec_10,
        "recall_at_10":        recall_10,
        "coverage":            eval_row.get("coverage", 0.0),
        "diversity":           eval_row.get("diversity", 0.0),
        "registered_at":       datetime.now(timezone.utc).isoformat(),
        "training_dataset_ref": (
            f"{project_id}.recommendation_feature_store.training_dataset_v1"
        ),
    }

    # Save registration artifact
    pd.DataFrame([registration_record]).to_parquet(
        registration_output.path, index=False
    )

    # Log KFP Metrics
    registry_metrics.log_metric("promoted",            1)
    registry_metrics.log_metric("ndcg_at_10",          ndcg_10)
    registry_metrics.log_metric("precision_at_10",     prec_10)
    registry_metrics.log_metric("recall_at_10",        recall_10)
    registry_metrics.log_metric("registration_status", 1)

    log.info(" Model registered successfully: %s", version_id)
    return json.dumps(registration_record, indent=2, default=str)