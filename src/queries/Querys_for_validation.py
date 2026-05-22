from enum import Enum
from config.feature_config import  Tables as T 
from config.feature_config import DataConfig

class Queries(Enum):
    """
    Centralized SQL Queries for Data Validation
    BigQuery - thelook_ecommerce Dataset
    """

    # 1) Missing Values Check
    MISSING_VALUES = f"""
    SELECT
      COUNTIF(user_id IS NULL) AS missing_users,
      COUNTIF(product_id IS NULL) AS missing_products,
      COUNTIF(sale_price IS NULL) AS missing_price
    FROM `{T.ORDER_ITEMS}`
    """

    # 2) Duplicate Records Check
    DUPLICATE_RECORDS = F"""
    SELECT
      order_id,
      product_id,
      user_id,
      COUNT(*) AS duplicates
    FROM `{T.ORDER_ITEMS}`
    GROUP BY order_id, product_id, user_id
    HAVING duplicates > 1
    """

    # 3) Negative Prices Check
    NEGATIVE_PRICES = f"""
    SELECT *
    FROM `{T.ORDER_ITEMS}`
    WHERE sale_price < 0
    """

    # 4) Invalid Ages Check
    INVALID_AGES = f"""
    SELECT *
    FROM `{T.USERS}`
    WHERE age < 0 OR age > 100
    """

    # 5) Broken Product Relations Check
    BROKEN_PRODUCT_RELATIONS = f"""
    SELECT DISTINCT oi.product_id
    FROM `{T.ORDER_ITEMS}` oi
    LEFT JOIN `{T.PRODUCTS}` p
    ON oi.product_id = p.id
    WHERE p.id IS NULL
    """

    # 6) Broken User Relations Check
    BROKEN_USER_RELATIONS = f"""
    SELECT DISTINCT oi.user_id
    FROM `{T.ORDER_ITEMS}` oi
    LEFT JOIN `{T.USERS}` u
    ON oi.user_id = u.id
    WHERE u.id IS NULL
    """

    # 7) Data Leakage Check (Future Data)
    DATA_LEAKAGE = f"""
    SELECT *
    FROM `{T.ORDERS}`
    WHERE created_at > CURRENT_TIMESTAMP()
    """

    # 8) Empty Product Names Check
    EMPTY_PRODUCT_NAMES = f"""
    SELECT *
    FROM `{T.PRODUCTS}`
    WHERE name IS NULL OR TRIM(name) = ''
    """

    # 9) Invalid Prices in Products
    INVALID_PRODUCT_PRICES = f"""
    SELECT *
    FROM `{T.PRODUCTS}`
    WHERE retail_price <= 0 OR cost < 0
    """

    # 10) Orders Without Items
    ORDERS_WITHOUT_ITEMS = f"""
    SELECT o.order_id
    FROM `{T.ORDERS}` o
    LEFT JOIN `{T.ORDER_ITEMS}` oi
    ON o.order_id = oi.order_id
    WHERE oi.order_id IS NULL
    """

  # 11) Schema Validation

    SCHEMA_VALIDATION = f"""
SELECT
    column_name,
    data_type
FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'order_items'
"""


    VALIDATED_ORDER_ITEMS = f"""
SELECT *
FROM `{T.ORDER_ITEMS}`
WHERE user_id IS NOT NULL
  AND product_id IS NOT NULL
  AND sale_price >= 0
"""