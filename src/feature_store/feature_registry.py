"""
Feature Registry
Centralised catalog of all features with metadata.

Each FeatureDefinition captures:
  - name, dtype, entity (user / product / interaction)
  - description & business rationale
  - freshness requirement (how often must it be recomputed?)
  - whether it is online-serving-critical or offline-only
  - version tracking

This registry drives:
  - Feature Store materialisation strategy
  - Automatic documentation generation
  - Schema validation before model training
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

#                                                           Enumerations 

class Entity(str, Enum):
    USER        = "user"
    PRODUCT     = "product"
    INTERACTION = "interaction"


class Freshness(str, Enum):
    REAL_TIME  = "real_time"   # must refresh within minutes
    HOURLY     = "hourly"
    DAILY      = "daily"
    WEEKLY     = "weekly"
    STATIC     = "static"      # changes only when catalogue is updated


class FeatureGroup(str, Enum):
    USER_DEMOGRAPHICS   = "user_demographics"
    USER_BEHAVIOUR      = "user_behaviour"
    USER_PREFERENCES    = "user_preferences"
    USER_ROLLING        = "user_rolling"
    PRODUCT_CATALOGUE   = "product_catalogue"
    PRODUCT_POPULARITY  = "product_popularity"
    PRODUCT_QUALITY     = "product_quality"
    INTERACTION         = "interaction"
    TEMPORAL            = "temporal"


#  Feature Definition Dataclass 

@dataclass
class FeatureDefinition:
    name:           str
    entity:         Entity
    group:          FeatureGroup
    dtype:          str                     # "float", "int", "string", "bool"
    description:    str
    freshness:      Freshness
    is_online:      bool                    # True → serve in online store
    version:        int = 1
    default_value:  Optional[float] = None  # for cold-start / missing values
    tags:           List[str] = field(default_factory=list)


#  Registry

class FeatureRegistry:
    """
    Single source of truth for all features in the recommendation system.
    """

    _features: List[FeatureDefinition] = []

    #  Registration 
    @classmethod
    def register(cls, *features: FeatureDefinition) -> None:
        cls._features.extend(features)

    @classmethod
    def get_all(cls) -> List[FeatureDefinition]:
        return list(cls._features)

    @classmethod
    def get_by_entity(cls, entity: Entity) -> List[FeatureDefinition]:
        return [f for f in cls._features if f.entity == entity]

    @classmethod
    def get_online_features(cls) -> List[FeatureDefinition]:
        return [f for f in cls._features if f.is_online]

    @classmethod
    def get_by_group(cls, group: FeatureGroup) -> List[FeatureDefinition]:
        return [f for f in cls._features if f.group == group]

    @classmethod
    def feature_names(cls, entity: Optional[Entity] = None) -> List[str]:
        features = cls.get_by_entity(entity) if entity else cls._features
        return [f.name for f in features]

    @classmethod
    def summary(cls) -> str:
        lines = ["Feature Registry Summary", "=" * 40]
        for entity in Entity:
            feats = cls.get_by_entity(entity)
            lines.append(f"\n[{entity.value.upper()}] — {len(feats)} features")
            for f in feats:
                lines.append(f"  • {f.name:40s} {f.dtype:8s} freshness={f.freshness.value}")
        return "\n".join(lines)


#  Register All Features 

#   User Demographics 
FeatureRegistry.register(
    FeatureDefinition("age",               Entity.USER, FeatureGroup.USER_DEMOGRAPHICS,
                      "int",    "User age in years",                        Freshness.STATIC,  True, default_value=30),
    FeatureDefinition("gender",            Entity.USER, FeatureGroup.USER_DEMOGRAPHICS,
                      "string", "User gender",                              Freshness.STATIC,  True),
    FeatureDefinition("country",           Entity.USER, FeatureGroup.USER_DEMOGRAPHICS,
                      "string", "User country of residence",                Freshness.STATIC,  True),
    FeatureDefinition("acquisition_source",Entity.USER, FeatureGroup.USER_DEMOGRAPHICS,
                      "string", "How the user was acquired (traffic source)",Freshness.STATIC, False),
    FeatureDefinition("account_age_days",  Entity.USER, FeatureGroup.USER_DEMOGRAPHICS,
                      "int",    "Days since account was created",           Freshness.DAILY,   True, default_value=0),
)

#  User Behaviour 
FeatureRegistry.register(
    FeatureDefinition("total_orders",             Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "int",   "Total completed orders",                  Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("total_spend",              Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "Cumulative spend in USD",                 Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("avg_order_value",          Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "Average value per order",                 Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("days_since_last_order",    Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "int",   "Recency: days since last purchase",       Freshness.DAILY, True,  default_value=9999),
    FeatureDefinition("return_rate",              Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "Fraction of items returned",              Freshness.DAILY, False, default_value=0.0),
    FeatureDefinition("purchase_frequency_per_day",Entity.USER,FeatureGroup.USER_BEHAVIOUR,
                      "float", "Orders per day since account creation",   Freshness.DAILY, False, default_value=0.0),
    FeatureDefinition("rfm_recency_score",        Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "RFM recency quintile (1-5)",              Freshness.DAILY, True,  default_value=1.0),
    FeatureDefinition("rfm_frequency_score",      Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "RFM frequency quintile (1-5)",            Freshness.DAILY, True,  default_value=1.0),
    FeatureDefinition("rfm_monetary_score",       Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "RFM monetary quintile (1-5)",             Freshness.DAILY, True,  default_value=1.0),
    FeatureDefinition("rfm_composite_score",      Entity.USER, FeatureGroup.USER_BEHAVIOUR,
                      "float", "Weighted composite RFM score",            Freshness.DAILY, True,  default_value=1.0),
)

# ── User Preferences
FeatureRegistry.register(
    FeatureDefinition("top_category_1", Entity.USER, FeatureGroup.USER_PREFERENCES,
                      "string", "Most purchased product category",    Freshness.WEEKLY, True),
    FeatureDefinition("top_category_2", Entity.USER, FeatureGroup.USER_PREFERENCES,
                      "string", "2nd most purchased category",        Freshness.WEEKLY, False),
    FeatureDefinition("top_category_3", Entity.USER, FeatureGroup.USER_PREFERENCES,
                      "string", "3rd most purchased category",        Freshness.WEEKLY, False),
)

# ── User Rolling Windows 
FeatureRegistry.register(
    FeatureDefinition("orders_last_7d",   Entity.USER, FeatureGroup.USER_ROLLING,
                      "int",   "Orders in last 7 days",   Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("orders_last_30d",  Entity.USER, FeatureGroup.USER_ROLLING,
                      "int",   "Orders in last 30 days",  Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("orders_last_90d",  Entity.USER, FeatureGroup.USER_ROLLING,
                      "int",   "Orders in last 90 days",  Freshness.DAILY, False, default_value=0),
    FeatureDefinition("spend_last_30d",   Entity.USER, FeatureGroup.USER_ROLLING,
                      "float", "Spend in last 30 days",   Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("is_active_30d",    Entity.USER, FeatureGroup.USER_ROLLING,
                      "bool",  "Had a purchase in last 30 days", Freshness.DAILY, True,  default_value=0),
)

# ── Product Catalogue 
FeatureRegistry.register(
    FeatureDefinition("category",      Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "string", "Product category",              Freshness.STATIC, True),
    FeatureDefinition("brand",         Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "string", "Product brand",                 Freshness.STATIC, True),
    FeatureDefinition("department",    Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "string", "Product department (M/F/Kids)", Freshness.STATIC, True),
    FeatureDefinition("retail_price",  Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "float",  "Listed retail price",           Freshness.STATIC, True,  default_value=0.0),
    FeatureDefinition("margin_rate",   Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "float",  "Gross margin as fraction",      Freshness.STATIC, False, default_value=0.0),
    FeatureDefinition("price_tier",    Entity.PRODUCT, FeatureGroup.PRODUCT_CATALOGUE,
                      "string", "Price bucket (budget/mid/premium/luxury)", Freshness.STATIC, True),
)

# ── Product Popularity 
FeatureRegistry.register(
    FeatureDefinition("total_purchases",         Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "int",   "Total number of purchases",                Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("unique_buyers",           Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "int",   "Number of distinct users who bought",      Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("popularity_score",        Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "float", "Normalised popularity within category",    Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("sales_last_30d",          Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "int",   "Sales in last 30 days",                    Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("sales_trend_30_vs_prev",  Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "float", "Sales ratio: last 30d vs prior 30-90d",    Freshness.DAILY, False, default_value=1.0),
    FeatureDefinition("log_total_purchases",     Entity.PRODUCT, FeatureGroup.PRODUCT_POPULARITY,
                      "float", "Log-transformed total purchases (for models)", Freshness.DAILY, False, default_value=0.0),
)

# ── Product Quality 
FeatureRegistry.register(
    FeatureDefinition("return_rate",      Entity.PRODUCT, FeatureGroup.PRODUCT_QUALITY,
                      "float", "Fraction of purchases that were returned", Freshness.WEEKLY, True,  default_value=0.0),
    FeatureDefinition("avg_discount_rate",Entity.PRODUCT, FeatureGroup.PRODUCT_QUALITY,
                      "float", "Average discount rate applied at sale",    Freshness.WEEKLY, False, default_value=0.0),
    FeatureDefinition("is_high_return",   Entity.PRODUCT, FeatureGroup.PRODUCT_QUALITY,
                      "bool",  "Return rate > 15%",                        Freshness.WEEKLY, True,  default_value=0),
    FeatureDefinition("is_new_product",   Entity.PRODUCT, FeatureGroup.PRODUCT_QUALITY,
                      "bool",  "Product first sold < 30 days ago",         Freshness.DAILY,  True,  default_value=0),
    FeatureDefinition("lifecycle_stage",  Entity.PRODUCT, FeatureGroup.PRODUCT_QUALITY,
                      "string","Product lifecycle (growing/stable/declining)", Freshness.WEEKLY, True),
)

# ── Interaction 
FeatureRegistry.register(
    FeatureDefinition("purchase_count",           Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "int",   "Times user purchased this product",     Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("view_count",               Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "int",   "Times user viewed this product",        Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("cart_count",               Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "int",   "Times user added to cart",              Freshness.DAILY, True,  default_value=0),
    FeatureDefinition("implicit_score",           Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "float", "Weighted engagement score",             Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("implicit_score_normalised",Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "float", "Implicit score normalised to [0,1]",    Freshness.DAILY, True,  default_value=0.0),
    FeatureDefinition("label",                    Entity.INTERACTION, FeatureGroup.INTERACTION,
                      "bool",  "Binary training label (purchased=1)",   Freshness.DAILY, False, default_value=0),
)