from dataclasses import dataclass

@dataclass
class DataConfig :
    expected_schema = {
    "id": "INTEGER",
    "order_id": "INTEGER",
    "user_id": "INTEGER",
    "product_id": "INTEGER",
    "inventory_item_id": "INTEGER",
    "status": "STRING",
    "created_at": "TIMESTAMP",
    "shipped_at": "TIMESTAMP",
    "delivered_at": "TIMESTAMP",
    "returned_at": "TIMESTAMP",
    "sale_price": "FLOAT",
}