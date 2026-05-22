import json
import os
from dataclasses import dataclass
from typing import Dict, List

from kfp import dsl
from kfp.dsl import Output, Dataset, Metrics
from dotenv import load_dotenv

load_dotenv()
GCP_PROJECT = os.environ.get("PROJECT_ID")
GCP_REPO = os.environ.get("REPO_NAME")
REGION = os.environ.get("REGION")
CUSTOM_BASE_IMAGE = f"{REGION}-docker.pkg.dev/{GCP_PROJECT}/{GCP_REPO}/xgbranker-pipeline-base:latest"


@dataclass
class DataValidationResult:
    passed: bool
    checks: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, int]


@dsl.component(
    base_image=CUSTOM_BASE_IMAGE
)
def data_validation(
    project_id: str,
    max_null_rate: int,
    metrics: Output[Metrics],
    validated_dataset: Output[Dataset],
) -> str:
    
    import json
    from src.utils.bq_utils import BigQueryManager
    from src.queries.Querys_for_validation import Queries
    from config.feature_config import DataConfig

 #===================================================================================================================   
    # 1. Initialize Containers & Client

    checks = {}
    errors = []
    warnings = []
    stats = {}
    validation_passed = True

    client = BigQueryManager(project_id=project_id)

 #===================================================================================================================   
    # 2. Execute Validation Checks


 #===================================================================================================================   

    # Check 1: Missing Values 
    
    missing_df = client.Query_To_Dataframe(query=Queries.MISSING_VALUES.value)
    missing_users = int(missing_df["missing_users"][0])
    missing_products = int(missing_df["missing_products"][0])
    missing_prices = int(missing_df["missing_price"][0])
    
    total_missing = missing_users + missing_products + missing_prices
    stats["missing_users"] = missing_users
    stats["missing_products"] = missing_products
    stats["missing_prices"] = missing_prices

    if total_missing > max_null_rate:
        validation_passed = False
        checks["missing_values"] = False
        errors.append(f"[Missing Values] Detected {total_missing} missing critical values (Limit: {max_null_rate}).")
    else:
        checks["missing_values"] = True


#===================================================================================================================   
    #  Check 2: Duplicate Records 
    duplicates_df = client.Query_To_Dataframe(query=Queries.DUPLICATE_RECORDS.value)
    stats["duplicate_records"] = len(duplicates_df)

    if not duplicates_df.empty:
        validation_passed = False
        checks["duplicate_records"] = False
        errors.append(f"[Duplicate Records] Found {len(duplicates_df)} duplicate rows in order_items.")
    else:
        checks["duplicate_records"] = True


#===================================================================================================================   
    #  Check 3: Negative Prices 
    negative_prices_df = client.Query_To_Dataframe(query=Queries.NEGATIVE_PRICES.value)
    stats["negative_prices"] = len(negative_prices_df)

    if not negative_prices_df.empty:
        validation_passed = False
        checks["negative_prices"] = False
        errors.append(f"[Negative Prices] Found {len(negative_prices_df)} negative sale prices in order_items.")
    else:
        checks["negative_prices"] = True


#===================================================================================================================   
    # Check 4: Invalid Ages 
    invalid_ages_df = client.Query_To_Dataframe(query=Queries.INVALID_AGES.value)
    stats["invalid_ages"] = len(invalid_ages_df)
    
    if not invalid_ages_df.empty:
        validation_passed = False
        checks["invalid_ages"] = False
        errors.append(f"[Invalid Ages] Found {len(invalid_ages_df)} users with unrealistic ages (<0 or >100).")
    else:
        checks["invalid_ages"] = True


#===================================================================================================================   
    # Check 5: Broken Product Relations 
    broken_product_df = client.Query_To_Dataframe(query=Queries.BROKEN_PRODUCT_RELATIONS.value)
    stats["broken_product_relations"] = len(broken_product_df)

    if not broken_product_df.empty:
        validation_passed = False
        checks["broken_product_relations"] = False
        errors.append(f"[Broken Relation] Found {len(broken_product_df)} order items linked to non-existent products.")
    else:
        checks["broken_product_relations"] = True


#===================================================================================================================   
    # Check 6: Broken User Relations 
    broken_user_df = client.Query_To_Dataframe(query=Queries.BROKEN_USER_RELATIONS.value)
    stats["broken_user_relations"] = len(broken_user_df)

    if not broken_user_df.empty:
        validation_passed = False
        checks["broken_user_relations"] = False
        errors.append(f"[Broken Relation] Found {len(broken_user_df)} order items linked to non-existent users.")
    else:
        checks["broken_user_relations"] = True


#===================================================================================================================   
    #  Check 7: Data Leakage (Future Data) 
    data_leakage_df = client.Query_To_Dataframe(query=Queries.DATA_LEAKAGE.value)
    stats["data_leakage"] = len(data_leakage_df)

    if not data_leakage_df.empty:
        validation_passed = False
        checks["data_leakage"] = False
        errors.append(f"[Data Leakage] Found {len(data_leakage_df)} records with creation dates in the future.")
    else:
        checks["data_leakage"] = True


#===================================================================================================================   
    #  Check 8: Empty Product Names 
    empty_product_names_df = client.Query_To_Dataframe(query=Queries.EMPTY_PRODUCT_NAMES.value)
    stats["empty_product_names"] = len(empty_product_names_df)

    if not empty_product_names_df.empty:
        
        validation_passed = False
        checks["empty_product_names"] = False
        errors.append(f"[Data Quality] Found {len(empty_product_names_df)} products with empty or NULL names.")
    else:
        checks["empty_product_names"] = True


#===================================================================================================================   
    # Check 9: Invalid Product Prices (Cost vs Retail) 
    invalid_prod_prices_df = client.Query_To_Dataframe(query=Queries.INVALID_PRODUCT_PRICES.value)
    stats["invalid_product_prices"] = len(invalid_prod_prices_df)

    if not invalid_prod_prices_df.empty:
        validation_passed = False
        checks["invalid_product_prices"] = False
        errors.append(f"[Invalid Prices] Found {len(invalid_prod_prices_df)} products with retail_price <= 0 or cost < 0.")
    else:
        checks["invalid_product_prices"] = True


#===================================================================================================================   
    # Check 10: Orders Without Items 
    orders_no_items_df = client.Query_To_Dataframe(query=Queries.ORDERS_WITHOUT_ITEMS.value)
    stats["orders_without_items"] = len(orders_no_items_df)

    if not orders_no_items_df.empty:
        validation_passed = False
        checks["orders_without_items"] = False
        errors.append(f"[Data Consistency] Found {len(orders_no_items_df)} orders containing zero items.")
    else:
        checks["orders_without_items"] = True


#===================================================================================================================   
    # Check 11: Schema Validation 
    schema_df = client.Query_To_Dataframe(query=Queries.SCHEMA_VALIDATION.value)
    actual_schema = dict(zip(schema_df["column_name"], schema_df["data_type"]))
    
    schema_errors = []
    for column, expected_type in DataConfig().expected_schema.items():

        if column not in actual_schema:
            schema_errors.append(f"Missing column: {column}")
            continue
            
        actual_type = actual_schema[column]
        if actual_type != expected_type:
            schema_errors.append(f"Column '{column}' expected {expected_type} but got {actual_type}")

    if schema_errors:
        validation_passed = False
        checks["schema_validation"] = False
        errors.extend([f"[Schema Error] {err}" for err in schema_errors])
    else:
        checks["schema_validation"] = True

    #===================================================================================================================   
    # 3. Export Validated Data & Log Metrics
    
    
    
    if validation_passed:
        validated_df = client.Query_To_Dataframe(query=Queries.VALIDATED_ORDER_ITEMS.value)
        validated_df.to_parquet(validated_dataset.path, index=False)
        stats["final_validated_records"] = len(validated_df)
    else:

        warnings.append("Validation failed: Validated dataset was not exported completely or safely.")

    # Logging Metrics to Vertex AI UI
    metrics.log_metric("total_missing_values", total_missing)
    metrics.log_metric("duplicate_records", stats["duplicate_records"])
    metrics.log_metric("broken_relations_total", stats["broken_product_relations"] + stats["broken_user_relations"])
    metrics.log_metric("validation_status", 1 if validation_passed else 0)

    #===================================================================================================================   
    # 4. Final Result Compilation
    
    result = DataValidationResult(
        passed=validation_passed,
        checks=checks,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )

    return json.dumps(result.__dict__, indent=2)