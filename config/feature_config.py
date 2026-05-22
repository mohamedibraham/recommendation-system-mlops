from dataclasses import dataclass, field
from typing import Dict, List


BQ_PROJECT = "bigquery-public-data"
BQ_DATASET = "thelook_ecommerce"
BQ_PREFIX  = f"`{BQ_PROJECT}.{BQ_DATASET}`"


class Tables:
    USERS               = f"{BQ_PREFIX}.users"
    PRODUCTS            = f"{BQ_PREFIX}.products"
    ORDERS              = f"{BQ_PREFIX}.orders"
    ORDER_ITEMS         = f"{BQ_PREFIX}.order_items"
    EVENTS              = f"{BQ_PREFIX}.events"
    INVENTORY_ITEMS     = f"{BQ_PREFIX}.inventory_items"
    DISTRIBUTION_CENTERS = f"{BQ_PREFIX}.distribution_centers"



@dataclass
class FeatureStoreConfig:
    project_id:        str
    dataset:           str = "recommendation_feature_store"

    
    user_features_table:        str = "user_features_v1"
    product_features_table:     str = "product_features_v1"
    interaction_features_table: str = "interaction_features_v1"
    temporal_features_table:    str = "temporal_features_v1"
    user_item_matrix_table:     str = "user_item_matrix_v1"
    training_dataset_table:     str = "training_dataset_v1"


    feature_version:            str = "v1"
    schema_version:             int = 1

    def table_ref(self, table_name: str) -> str:
        return f"{self.project_id}.{self.dataset}.{table_name}"




ROLLING_WINDOWS: List[int] = [7, 30, 90, 365]



INTERACTION_WEIGHTS: Dict[str, float] = {
    "purchase":  5.0,
    "cart":      3.0,
    "product":   1.0,   
    "department": 0.5,
    "home":       0.1,
    "cancel":    -1.0,
}


@dataclass
class DataConfig:
    expected_schema: Dict[str, str] = field(default_factory=lambda: {
        "id":          "INTEGER",
        "order_id":    "INTEGER",
        "user_id":     "INTEGER",
        "product_id":  "INTEGER",
        "sale_price":  "FLOAT",
        "status":      "STRING",
        "created_at":  "TIMESTAMP",
        "returned_at": "TIMESTAMP",
        "shipped_at":  "TIMESTAMP",
        "delivered_at": "TIMESTAMP",
        "inventory_item_id": "INTEGER",
    })