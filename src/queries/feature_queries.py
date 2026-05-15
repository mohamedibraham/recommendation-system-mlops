"""
Feature Engineering SQL Queries
All queries for computing user, product, interaction, and temporal features
from BigQuery Public Dataset: thelook_ecommerce.

Design principles:
- Each query is self-contained and runnable independently.
- Point-in-Time safe: uses AS OF TIMESTAMP parameter where needed.
- Optimised for BigQuery: partition pruning via created_at filters.
- No data leakage: all aggregations respect the reference_date boundary.
"""

from enum import Enum
from config.feature_config import Tables as T


class UserFeatureQueries(Enum):
    """SQL queries for computing user-level features."""

    # Core User Demographics & Purchase Behaviour 
    USER_CORE_FEATURES = f"""
    SELECT
        u.id                                                   AS user_id,
        u.age,
        u.gender,
        u.country,
        u.state,
        u.city,
        u.traffic_source                                       AS acquisition_source,

        -- Account tenure
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), u.created_at, DAY) AS account_age_days,

        -- Purchase volume
        COUNT(DISTINCT o.order_id)                             AS total_orders,
        COUNT(DISTINCT oi.id)                                  AS total_items_purchased,
        COUNT(DISTINCT oi.product_id)                          AS unique_products_purchased,
        COUNT(DISTINCT p.category)                             AS unique_categories_purchased,
        COUNT(DISTINCT p.brand)                                AS unique_brands_purchased,

        -- Spend metrics
        SUM(oi.sale_price)                                     AS total_spend,
        AVG(oi.sale_price)                                     AS avg_item_price,
        MAX(oi.sale_price)                                     AS max_item_price,
        MIN(oi.sale_price)                                     AS min_item_price,
        STDDEV(oi.sale_price)                                  AS stddev_item_price,

        -- Recency
        MAX(o.created_at)                                      AS last_order_date,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(o.created_at), DAY) AS days_since_last_order,
        MIN(o.created_at)                                      AS first_order_date,

        -- Purchase frequency (orders per active day)
        SAFE_DIVIDE(
            COUNT(DISTINCT o.order_id),
            NULLIF(TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), u.created_at, DAY), 0)
        )                                                      AS purchase_frequency_per_day,

        -- Return behaviour (negative signal)
        COUNTIF(oi.status = 'Returned')                        AS total_returns,
        SAFE_DIVIDE(
            COUNTIF(oi.status = 'Returned'),
            NULLIF(COUNT(oi.id), 0)
        )                                                      AS return_rate,

        -- Average order value
        SAFE_DIVIDE(SUM(oi.sale_price), NULLIF(COUNT(DISTINCT o.order_id), 0)) AS avg_order_value

    FROM {T.USERS} u
    LEFT JOIN {T.ORDERS} o
           ON u.id = o.user_id
          AND o.status IN ('Complete', 'Shipped', 'Processing')
    LEFT JOIN {T.ORDER_ITEMS} oi
           ON o.order_id = oi.order_id
    LEFT JOIN {T.PRODUCTS} p
           ON oi.product_id = p.id
    GROUP BY
        u.id, u.age, u.gender, u.country, u.state,
        u.city, u.traffic_source, u.created_at
    """

    #  User Top Category Preferences 
    USER_CATEGORY_PREFERENCES = f"""
    WITH category_stats AS (
        SELECT
            oi.user_id,
            p.category,
            COUNT(*)            AS purchase_count,
            SUM(oi.sale_price)  AS category_spend
        FROM {T.ORDER_ITEMS} oi
        JOIN {T.PRODUCTS} p ON oi.product_id = p.id
        WHERE oi.status = 'Complete'
        GROUP BY oi.user_id, p.category
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchase_count DESC) AS rnk
        FROM category_stats
    )
    SELECT
        user_id,
        MAX(IF(rnk = 1, category, NULL))       AS top_category_1,
        MAX(IF(rnk = 2, category, NULL))       AS top_category_2,
        MAX(IF(rnk = 3, category, NULL))       AS top_category_3,
        MAX(IF(rnk = 1, purchase_count, NULL)) AS top_cat_1_count,
        MAX(IF(rnk = 2, purchase_count, NULL)) AS top_cat_2_count,
        MAX(IF(rnk = 3, purchase_count, NULL)) AS top_cat_3_count
    FROM ranked
    GROUP BY user_id
    """

    #  Rolling Window Features (7 / 30 / 90 days) 
    USER_ROLLING_FEATURES = f"""
    SELECT
        oi.user_id,

        -- 7-day window
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
                                                             AS orders_last_7d,
        SUM(IF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY),
               oi.sale_price, 0))                           AS spend_last_7d,

        -- 30-day window
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY))
                                                             AS orders_last_30d,
        SUM(IF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY),
               oi.sale_price, 0))                           AS spend_last_30d,

        -- 90-day window
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY))
                                                             AS orders_last_90d,
        SUM(IF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY),
               oi.sale_price, 0))                           AS spend_last_90d,

        -- 365-day window
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY))
                                                             AS orders_last_365d,
        SUM(IF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY),
               oi.sale_price, 0))                           AS spend_last_365d

    FROM {T.ORDER_ITEMS} oi
    JOIN {T.ORDERS} o ON oi.order_id = o.order_id
    WHERE o.status IN ('Complete', 'Shipped')
    GROUP BY oi.user_id
    """

    #  User Behavioural Event Features 
    USER_EVENT_FEATURES = f"""
    SELECT
        user_id,
        COUNT(DISTINCT session_id)                         AS total_sessions,
        COUNT(*)                                           AS total_events,
        COUNTIF(event_type = 'product')                   AS product_views,
        COUNTIF(event_type = 'cart')                      AS cart_events,
        COUNTIF(event_type = 'purchase')                  AS purchase_events,
        COUNTIF(event_type = 'cancel')                    AS cancel_events,
        COUNTIF(event_type = 'department')                AS department_views,

        -- Conversion signals
        SAFE_DIVIDE(
            COUNTIF(event_type = 'purchase'),
            NULLIF(COUNTIF(event_type = 'product'), 0)
        )                                                  AS view_to_purchase_rate,

        SAFE_DIVIDE(
            COUNTIF(event_type = 'purchase'),
            NULLIF(COUNTIF(event_type = 'cart'), 0)
        )                                                  AS cart_to_purchase_rate,

        -- Recency of engagement
        MAX(created_at)                                    AS last_event_at,
        TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(created_at), DAY) AS days_since_last_event,

        -- Session depth
        SAFE_DIVIDE(COUNT(*), NULLIF(COUNT(DISTINCT session_id), 0)) AS avg_events_per_session,

        -- Traffic source from events
        APPROX_TOP_COUNT(traffic_source, 1)[OFFSET(0)].value AS dominant_traffic_source

    FROM {T.EVENTS}
    GROUP BY user_id
    """


class ProductFeatureQueries(Enum):
    """SQL queries for computing product-level features."""

    #  Core Product Features 
    PRODUCT_CORE_FEATURES = f"""
    SELECT
        p.id                                              AS product_id,
        p.name                                            AS product_name,
        p.category,
        p.brand,
        p.department,
        p.retail_price,
        p.cost,

        -- Margin
        p.retail_price - p.cost                           AS gross_margin,
        SAFE_DIVIDE(p.retail_price - p.cost,
                    NULLIF(p.retail_price, 0))            AS margin_rate,

        -- Popularity metrics
        COUNT(DISTINCT oi.user_id)                        AS unique_buyers,
        COUNT(oi.id)                                      AS total_purchases,
        SUM(oi.sale_price)                                AS total_revenue,

        -- Pricing signals
        AVG(oi.sale_price)                                AS avg_sale_price,
        p.retail_price - AVG(oi.sale_price)               AS avg_discount_amount,
        SAFE_DIVIDE(
            p.retail_price - AVG(oi.sale_price),
            NULLIF(p.retail_price, 0)
        )                                                 AS avg_discount_rate,

        -- Quality signals
        SAFE_DIVIDE(
            COUNTIF(oi.status = 'Returned'),
            NULLIF(COUNT(oi.id), 0)
        )                                                 AS return_rate,

        -- Availability window
        MIN(oi.created_at)                                AS first_purchased_at,
        MAX(oi.created_at)                                AS last_purchased_at,
        TIMESTAMP_DIFF(MAX(oi.created_at), MIN(oi.created_at), DAY)
                                                          AS active_days

    FROM {T.PRODUCTS} p
    LEFT JOIN {T.ORDER_ITEMS} oi ON p.id = oi.product_id
    GROUP BY p.id, p.name, p.category, p.brand,
             p.department, p.retail_price, p.cost
    """

    #  Product Popularity Score (normalised within category) 
    PRODUCT_POPULARITY_SCORE = f"""
    WITH raw AS (
        SELECT
            p.id          AS product_id,
            p.category,
            COUNT(oi.id)  AS purchase_count,
            COUNT(DISTINCT oi.user_id) AS unique_buyers
        FROM {T.PRODUCTS} p
        LEFT JOIN {T.ORDER_ITEMS} oi
               ON p.id = oi.product_id
              AND oi.status = 'Complete'
        GROUP BY p.id, p.category
    ),
    cat_stats AS (
        SELECT
            category,
            MAX(purchase_count)   AS cat_max_purchases,
            MAX(unique_buyers)    AS cat_max_buyers
        FROM raw
        GROUP BY category
    )
    SELECT
        r.product_id,
        r.category,
        r.purchase_count,
        r.unique_buyers,
        -- Normalised 0-1 popularity within category
        SAFE_DIVIDE(r.purchase_count, NULLIF(cs.cat_max_purchases, 0)) AS popularity_score,
        SAFE_DIVIDE(r.unique_buyers,  NULLIF(cs.cat_max_buyers, 0))   AS reach_score
    FROM raw r
    JOIN cat_stats cs USING (category)
    """

    #  Product Rolling Sales (recency signal) 
    PRODUCT_ROLLING_SALES = f"""
    SELECT
        oi.product_id,

        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
                                                           AS sales_last_7d,
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY))
                                                           AS sales_last_30d,
        COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY))
                                                           AS sales_last_90d,

        -- Trend: recent vs historical
        SAFE_DIVIDE(
            COUNTIF(o.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)),
            NULLIF(COUNTIF(
                o.created_at BETWEEN
                    TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
                    AND TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            ), 0)
        )                                                  AS sales_trend_30_vs_prev

    FROM {T.ORDER_ITEMS} oi
    JOIN {T.ORDERS} o ON oi.order_id = o.order_id
    WHERE o.status IN ('Complete', 'Shipped')
    GROUP BY oi.product_id
    """

    #  Co-purchase features (products bought together) 
    PRODUCT_CO_PURCHASE_STATS = f"""
    SELECT
        oi1.product_id,
        COUNT(DISTINCT oi1.order_id) AS orders_with_co_purchase,
        COUNT(DISTINCT oi2.product_id) AS unique_co_purchased_products
    FROM {T.ORDER_ITEMS} oi1
    JOIN {T.ORDER_ITEMS} oi2
      ON oi1.order_id = oi2.order_id
     AND oi1.product_id != oi2.product_id
    WHERE oi1.status = 'Complete' AND oi2.status = 'Complete'
    GROUP BY oi1.product_id
    """


class InteractionFeatureQueries(Enum):
    """SQL queries for user-item interaction matrix and implicit feedback."""

    #  Implicit Feedback Matrix 
    # Combines purchase history + event signals (views, cart adds) into
    # a single interaction score per (user, product) pair.
    USER_ITEM_IMPLICIT_FEEDBACK = f"""
    WITH purchases AS (
        SELECT
            oi.user_id,
            oi.product_id,
            COUNT(*)           AS purchase_count,
            SUM(oi.sale_price) AS total_spend,
            MAX(o.created_at)  AS last_purchased_at
        FROM {T.ORDER_ITEMS} oi
        JOIN {T.ORDERS} o ON oi.order_id = o.order_id
        WHERE oi.status = 'Complete'
        GROUP BY oi.user_id, oi.product_id
    ),
    event_signals AS (
        -- Extract product_id from event URI pattern: /products/12345
        SELECT
            user_id,
            CAST(REGEXP_EXTRACT(uri, r'/products/(\\d+)') AS INT64) AS product_id,
            COUNTIF(event_type = 'product') AS view_count,
            COUNTIF(event_type = 'cart')    AS cart_count
        FROM {T.EVENTS}
        WHERE REGEXP_CONTAINS(uri, r'/products/\\d+')
        GROUP BY user_id, product_id
    )
    SELECT
        COALESCE(p.user_id, e.user_id)       AS user_id,
        COALESCE(p.product_id, e.product_id) AS product_id,

        -- Raw interaction counts
        COALESCE(p.purchase_count, 0)        AS purchase_count,
        COALESCE(e.view_count, 0)            AS view_count,
        COALESCE(e.cart_count, 0)            AS cart_count,
        COALESCE(p.total_spend, 0.0)         AS total_spend,
        p.last_purchased_at,

        -- Composite implicit feedback score:
        --   purchase=5, cart=3, view=1  (from INTERACTION_WEIGHTS config)
        COALESCE(p.purchase_count, 0) * 5.0
          + COALESCE(e.cart_count,  0) * 3.0
          + COALESCE(e.view_count,  0) * 1.0
                                             AS implicit_score

    FROM purchases p
    FULL OUTER JOIN event_signals e
               ON p.user_id = e.user_id
              AND p.product_id = e.product_id
    """

    #  Session-Level Context Features 
    SESSION_CONTEXT_FEATURES = f"""
    SELECT
        user_id,
        session_id,
        MIN(created_at)                                      AS session_start,
        MAX(created_at)                                      AS session_end,
        TIMESTAMP_DIFF(MAX(created_at), MIN(created_at), SECOND) AS session_duration_sec,
        COUNT(*)                                             AS events_in_session,
        COUNTIF(event_type = 'product')                      AS product_views_in_session,
        COUNTIF(event_type = 'cart')                         AS cart_adds_in_session,
        COUNTIF(event_type = 'purchase')                     AS purchases_in_session,
        MAX(traffic_source)                                  AS traffic_source,
        MAX(browser)                                         AS browser
    FROM {T.EVENTS}
    GROUP BY user_id, session_id
    """

    #  Negative Feedback Signals 
    NEGATIVE_FEEDBACK = f"""
    SELECT
        oi.user_id,
        oi.product_id,
        COUNT(*)                                             AS return_count,
        MAX(oi.returned_at)                                  AS last_returned_at
    FROM {T.ORDER_ITEMS} oi
    WHERE oi.status = 'Returned'
      AND oi.returned_at IS NOT NULL
    GROUP BY oi.user_id, oi.product_id
    """


class TemporalFeatureQueries(Enum):
    """Queries producing time-based and seasonal features."""

    #  User Purchase Seasonality 
    USER_PURCHASE_SEASONALITY = f"""
    SELECT
        oi.user_id,

        -- Day-of-week distribution
        COUNTIF(EXTRACT(DAYOFWEEK FROM o.created_at) IN (1,7)) AS weekend_purchases,
        COUNTIF(EXTRACT(DAYOFWEEK FROM o.created_at) NOT IN (1,7)) AS weekday_purchases,

        -- Hour-of-day (UTC)
        AVG(EXTRACT(HOUR FROM o.created_at))                AS avg_purchase_hour,
        APPROX_TOP_COUNT(EXTRACT(HOUR FROM o.created_at), 1)[OFFSET(0)].value
                                                            AS peak_purchase_hour,

        -- Month distribution
        APPROX_TOP_COUNT(EXTRACT(MONTH FROM o.created_at), 1)[OFFSET(0)].value
                                                            AS peak_purchase_month,

        -- Quarter
        APPROX_TOP_COUNT(EXTRACT(QUARTER FROM o.created_at), 1)[OFFSET(0)].value
                                                            AS peak_purchase_quarter,

        -- Purchase cadence (inter-purchase interval in days)
        AVG(
            TIMESTAMP_DIFF(
                o.created_at,
                LAG(o.created_at) OVER (PARTITION BY oi.user_id ORDER BY o.created_at),
                DAY
            )
        )                                                   AS avg_days_between_orders

    FROM {T.ORDER_ITEMS} oi
    JOIN {T.ORDERS} o ON oi.order_id = o.order_id
    WHERE o.status IN ('Complete', 'Shipped')
    GROUP BY oi.user_id
    """

    #  Product Seasonality 
    PRODUCT_SEASONALITY = f"""
    SELECT
        oi.product_id,
        EXTRACT(MONTH FROM o.created_at)   AS month,
        EXTRACT(QUARTER FROM o.created_at) AS quarter,
        COUNT(*)                           AS sales_count,
        SUM(oi.sale_price)                 AS revenue
    FROM {T.ORDER_ITEMS} oi
    JOIN {T.ORDERS} o ON oi.order_id = o.order_id
    WHERE o.status = 'Complete'
    GROUP BY oi.product_id, month, quarter
    """


class FeatureStoreQueries(Enum):
    """DDL and DML for the Feature Store tables in BigQuery."""

    CREATE_FEATURE_STORE_DATASET = """
    CREATE SCHEMA IF NOT EXISTS `{project_id}.recommendation_feature_store`
    OPTIONS (
        description = 'Offline feature store for recommendation system',
        location = 'US'
    )
    """

    CREATE_USER_FEATURES_TABLE = """
    CREATE OR REPLACE TABLE `{project_id}.recommendation_feature_store.user_features_v1`
    PARTITION BY DATE(feature_timestamp)
    CLUSTER BY user_id
    OPTIONS (
        description = 'User features — version 1',
        labels = [('feature_version', 'v1'), ('entity', 'user')]
    )
    AS
    SELECT *, CURRENT_TIMESTAMP() AS feature_timestamp
    FROM ({user_features_query})
    """

    CREATE_PRODUCT_FEATURES_TABLE = """
    CREATE OR REPLACE TABLE `{project_id}.recommendation_feature_store.product_features_v1`
    PARTITION BY DATE(feature_timestamp)
    CLUSTER BY product_id
    OPTIONS (
        description = 'Product features — version 1',
        labels = [('feature_version', 'v1'), ('entity', 'product')]
    )
    AS
    SELECT *, CURRENT_TIMESTAMP() AS feature_timestamp
    FROM ({product_features_query})
    """

    # Point-in-Time correct training dataset
    CREATE_TRAINING_DATASET = """
    SELECT
        uim.*,
        uf.* EXCEPT (user_id, feature_timestamp),
        pf.* EXCEPT (product_id, feature_timestamp)
    FROM `{project_id}.recommendation_feature_store.user_item_matrix_v1` uim
    LEFT JOIN `{project_id}.recommendation_feature_store.user_features_v1` uf
           ON uim.user_id = uf.user_id
    LEFT JOIN `{project_id}.recommendation_feature_store.product_features_v1` pf
           ON uim.product_id = pf.product_id
    """