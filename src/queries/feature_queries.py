from enum import Enum
from config.feature_config import Tables as T

class Feature_Queries(Enum):

    session_context = f"""
session_context AS (
  SELECT 
    e.session_id,
    e.user_id,
    e.created_at AS session_start_time,
    p.category AS current_session_category,
    e.product_id AS interacted_product_id,
    CASE 
      WHEN e.event_type IN ('cart', 'purchase') THEN 1 
      ELSE 0 
    END AS is_positive
  FROM 
    `{T.EVENTS}` e
  JOIN 
    `{T.PRODUCTS}` p ON e.product_id = p.id
  WHERE 
    e.session_id IS NOT NULL
)"""
    
    top_products_per_category = f"""
top_products_per_category AS (
  SELECT p.id, p.category,
    COUNT(oi.id) as sales_count
  FROM `{T.PRODUCTS}` p
  LEFT JOIN `{T.ORDER_ITEMS}` oi 
    ON p.id = oi.product_id
    AND oi.created_at < sc.session_start_time
  GROUP BY 
    p.id, p.category
)"""
    
    positive_samples = """
positive_samples AS (
  SELECT DISTINCT 
    session_id, 
    user_id, 
    current_session_category, 
    interacted_product_id AS product_id, 
    1 AS label
  FROM 
    session_context
  WHERE 
    is_positive = 1
)"""
    
    candidate_generation = """
candidate_generation AS (
  SELECT 
    p.session_id,
    p.user_id,
    p.current_session_category,
    t.candidate_product_id AS product_id,
    0 AS label
  FROM 
    positive_samples p
  JOIN 
    top_products_per_category t ON p.current_session_category = t.category
  WHERE 
    t.product_rank <= 20 
)"""
    
    final_deduped = """
combined_candidates AS (
  SELECT * FROM positive_samples
  UNION DISTINCT
  SELECT * FROM candidate_generation
),

final_deduped AS (
  SELECT 
    session_id,
    user_id,
    product_id,
    MAX(label) AS target_label 
  FROM 
    combined_candidates
  GROUP BY 
    session_id, user_id, product_id
)"""
    
    Feature_Attachment = f"""
SELECT 
  c.session_id AS qid, 
  c.user_id,
  c.product_id,
  u.age AS user_age,
  u.gender AS user_gender,
  u.country AS user_country,
  p.retail_price,
  p.cost,
  (p.retail_price - p.cost) AS profit_margin,
  p.brand,
  c.target_label,
  st.session_start_time
FROM 
  final_deduped c
JOIN 
  `{T.USERS}` u ON c.user_id = u.id
JOIN 
  `{T.PRODUCTS}` p ON c.product_id = p.id
LEFT JOIN 
  (SELECT session_id, MIN(session_start_time) as session_start_time FROM session_context GROUP BY session_id) st
ON 
  c.session_id = st.session_id
ORDER BY 
  st.session_start_time, qid;
"""

    @classmethod
    def get_full_query(cls) -> str:

        ctes = ",\n".join([
            cls.session_context.value,
            cls.top_products_per_category.value,
            cls.positive_samples.value,
            cls.candidate_generation.value,
            cls.final_deduped.value
        ])
        
        full_sql = f"WITH \n{ctes}\n{cls.Feature_Attachment.value}"
        return full_sql