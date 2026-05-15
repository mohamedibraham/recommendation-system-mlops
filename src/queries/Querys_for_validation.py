from enum import Enum


class Queries(Enum):
    """
    Centralized SQL Queries for Data Validation
    BigQuery - thelook_ecommerce Dataset
    """

    # 1) Missing Values Check
    MISSING_VALUES = """
    SELECT
      COUNTIF(user_id IS NULL) AS missing_users,
      COUNTIF(product_id IS NULL) AS missing_products,
      COUNTIF(sale_price IS NULL) AS missing_price
    FROM `bigquery-public-data.thelook_ecommerce.order_items`
    """

    # 2) Duplicate Records Check
    DUPLICATE_RECORDS = """
    SELECT
      order_id,
      product_id,
      user_id,
      COUNT(*) AS duplicates
    FROM `bigquery-public-data.thelook_ecommerce.order_items`
    GROUP BY order_id, product_id, user_id
    HAVING duplicates > 1
    """

    # 3) Negative Prices Check
    NEGATIVE_PRICES = """
    SELECT *
    FROM `bigquery-public-data.thelook_ecommerce.order_items`
    WHERE sale_price < 0
    """

    # 4) Invalid Ages Check
    INVALID_AGES = """
    SELECT *
    FROM `bigquery-public-data.thelook_ecommerce.users`
    WHERE age < 0 OR age > 100
    """

    # 5) Broken Product Relations Check
    BROKEN_PRODUCT_RELATIONS = """
    SELECT DISTINCT oi.product_id
    FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
    LEFT JOIN `bigquery-public-data.thelook_ecommerce.products` p
    ON oi.product_id = p.id
    WHERE p.id IS NULL
    """

    # 6) Broken User Relations Check
    BROKEN_USER_RELATIONS = """
    SELECT DISTINCT oi.user_id
    FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
    LEFT JOIN `bigquery-public-data.thelook_ecommerce.users` u
    ON oi.user_id = u.id
    WHERE u.id IS NULL
    """

    # 7) Data Leakage Check (Future Data)
    DATA_LEAKAGE = """
    SELECT *
    FROM `bigquery-public-data.thelook_ecommerce.orders`
    WHERE created_at > CURRENT_TIMESTAMP()
    """

    # 8) Empty Product Names Check
    EMPTY_PRODUCT_NAMES = """
    SELECT *
    FROM `bigquery-public-data.thelook_ecommerce.products`
    WHERE name IS NULL OR TRIM(name) = ''
    """

    # 9) Invalid Prices in Products
    INVALID_PRODUCT_PRICES = """
    SELECT *
    FROM `bigquery-public-data.thelook_ecommerce.products`
    WHERE retail_price <= 0 OR cost < 0
    """

    # 10) Orders Without Items
    ORDERS_WITHOUT_ITEMS = """
    SELECT o.order_id
    FROM `bigquery-public-data.thelook_ecommerce.orders` o
    LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
    ON o.order_id = oi.order_id
    WHERE oi.order_id IS NULL
    """

  # 11) Schema Validation

    SCHEMA_VALIDATION = """
SELECT
    column_name,
    data_type
FROM `bigquery-public-data.thelook_ecommerce.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'order_items'
"""


    VALIDATED_ORDER_ITEMS = """
SELECT *
FROM `bigquery-public-data.thelook_ecommerce.order_items`
WHERE user_id IS NOT NULL
  AND product_id IS NOT NULL
  AND sale_price >= 0
"""