from dataclasses import dataclass
from typing import Dict, List

from kfp import dsl
from kfp.dsl import Output, Dataset, Metrics

from src.utils.bq_utils import BigQueryManager
from src.queries.Querys_for_validation import Queries
from config.config import DataConfig


@dataclass
class DataValidationResult:
    passed: bool
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, int]


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "google-cloud-bigquery==3.11.4",
        "pyarrow==13.0.0",
    ],
)

def data_validation(
    project_id: str,
    max_null_rate: int,
    metrics: Output[Metrics],
    validated_dataset: Output[Dataset],
) -> str:

    import json

    # Initialize result containers
    checks = {}
    errors = []
    warnings = []
    stats = {}

    validation_passed = True

    # BigQuery Client

    client = BigQueryManager(project_id=project_id)

#=======================================================================================================================
    # 1) Missing Values Check
    
    missing_df = client.Query_To_Dataframe(
        query=Queries.MISSING_VALUES.value
    )

    missing_users = int(missing_df["missing_users"][0])
    missing_products = int(missing_df["missing_products"][0])
    missing_prices = int(missing_df["missing_price"][0])

    total_missing = (
        missing_users +
        missing_products +
        missing_prices
    )

    stats["missing_users"] = missing_users
    stats["missing_products"] = missing_products
    stats["missing_prices"] = missing_prices

    if total_missing > max_null_rate:
        validation_passed = False

        checks["missing_values"] = False

        errors.append(
            f"Missing values detected: {total_missing}"
        )

    else:
        checks["missing_values"] = True

#===================================================================================================================
    # 2) Duplicate Records Check

    duplicates_df = client.Query_To_Dataframe(
        query=Queries.DUPLICATE_RECORDS.value
    )

    duplicate_count = len(duplicates_df)

    stats["duplicate_records"] = duplicate_count

    if duplicate_count > 0:
        validation_passed = False

        checks["duplicate_records"] = False

        errors.append(
            f"Found {duplicate_count} duplicate records"
        )

    else:
        checks["duplicate_records"] = True

#==========================================================================================================
    # 3) Negative Prices Check

    negative_prices_df = client.Query_To_Dataframe(
        query=Queries.NEGATIVE_PRICES.value
    )

    negative_prices_count = len(negative_prices_df)

    stats["negative_prices"] = negative_prices_count

    if negative_prices_count > 0:

        validation_passed = False

        checks["negative_prices"] = False

        errors.append(
            f"Found {negative_prices_count} negative prices"
        )

    else:
        checks["negative_prices"] = True


#=================================================================================================
    # 4) Schema Validation

    schema_df = client.Query_To_Dataframe(
           query=Queries.SCHEMA_VALIDATION.value
        )

    actual_schema = dict(
        zip(
        schema_df["column_name"],
        schema_df["data_type"]
            )
    )

    schema_errors = []

    for column, expected_type in DataConfig.expected_schema.items():

         # Missing Column
         if column not in actual_schema:

                schema_errors.append(
                f"Missing column: {column}"
                )

                continue

         # Wrong Data Type
         actual_type = actual_schema[column]

         if actual_type != expected_type:

                schema_errors.append(
                f"Column '{column}' expected "
                f"{expected_type} but got {actual_type}"
                )

#===================================================================================================
    # Final Schema Validation Result

    if schema_errors:

            validation_passed = False

            checks["schema_validation"] = False

            errors.extend(schema_errors)

    else:

             checks["schema_validation"] = True

    

    # =====================================================
    # Save Validated Dataset

    validated_df = client.Query_To_Dataframe(
    query=Queries.VALIDATED_ORDER_ITEMS.value
        )

    validated_df.to_parquet(
            validated_dataset.path,
             index=False
           )
    

    # Final Validation Result

    result = DataValidationResult(
        passed=validation_passed,
        checks=checks,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )

    result_json = json.dumps(
        result.__dict__,
        indent=2
    )
    metrics.log_metric("missing_users", missing_users)
    
    return result_json
    

    