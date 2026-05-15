import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

logger = logging.getLogger(__name__)


# Core Metric Functions


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at K.

    DCG@K = Σ (2^rel_i - 1) / log2(i + 2)   for i in 0..k-1
    """
    relevances = np.asarray(relevances[:k], dtype=np.float64)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1, dtype=np.float64)
    return float(np.sum((2 ** relevances - 1) / np.log2(positions + 1)))


def ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Normalised DCG at K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant:    Set of ground-truth relevant item IDs.
        k:           Cutoff.

    Returns:
        NDCG@K ∈ [0, 1]
    """
    if not relevant:
        return 0.0
    gains   = np.array([1.0 if item in relevant else 0.0
                        for item in recommended[:k]])
    ideal   = np.ones(min(len(relevant), k), dtype=np.float64)
    ideal_dcg   = dcg_at_k(ideal, k)
    actual_dcg  = dcg_at_k(gains, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """P@K = |recommended ∩ relevant| / K"""
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    """R@K = |recommended ∩ relevant| / |relevant|"""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: List, relevant: set, k: int) -> float:
    """HR@K = 1 if any recommended item is relevant, else 0."""
    return float(any(item in relevant for item in recommended[:k]))


def average_precision_at_k(recommended: List, relevant: set, k: int) -> float:
    """
    Average Precision at K.
    AP@K = (1/|relevant|) * Σ P@i * rel(i)   for i in 1..k
    """
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits  += 1
            score += hits / i
    return score / min(len(relevant), k)


def mean_reciprocal_rank(recommended: List, relevant: set) -> float:
    """MRR = 1 / rank_of_first_relevant_item"""
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0



# Beyond-Accuracy Metrics

def catalogue_coverage(
    recommendations: Dict[int, List],
    total_items:     int,
) -> float:
    """
    Fraction of the product catalogue that appears in at least one
    recommendation list.

    Coverage = |∪ recommended_items| / total_items
    """
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def intra_list_diversity(
    recommendations: Dict[int, List],
    item_categories: Dict[int, str],
    k:               int = 10,
) -> float:
    """
    Average fraction of distinct categories within each recommendation list.

    ILD = (1 / |users|) * Σ (unique_categories / k)
    """
    diversities = []
    for recs in recommendations.values():
        cats = [item_categories.get(item, "unknown") for item in recs[:k]]
        if len(cats) > 0:
            diversities.append(len(set(cats)) / len(cats))
    return float(np.mean(diversities)) if diversities else 0.0


def novelty(
    recommendations: Dict[int, List],
    item_popularity: Dict[int, int],   # product_id → total_purchase_count
    k:               int = 10,
) -> float:
    """
    Average log-popularity of recommended items.
    Lower novelty = more popular items; higher novelty = more long-tail.

    Novelty@K = -mean( log2( popularity(item) + 1 ) )
    """
    scores = []
    for recs in recommendations.values():
        for item in recs[:k]:
            pop = item_popularity.get(item, 1)
            scores.append(np.log2(pop + 1))
    # Return as self-info: higher = more novel (less popular)
    return -float(np.mean(scores)) if scores else 0.0


def popular_item_bias(
    recommendations: Dict[int, List],
    item_popularity: Dict[int, int],
    popularity_threshold_percentile: float = 80.0,
    k: int = 10,
) -> float:
    """
    Fraction of recommended items that are in the top popularity percentile.
    A high score indicates the model over-recommends popular items.
    """
    threshold = np.percentile(list(item_popularity.values()),
                              popularity_threshold_percentile)
    popular_hits, total = 0, 0
    for recs in recommendations.values():
        for item in recs[:k]:
            total += 1
            if item_popularity.get(item, 0) >= threshold:
                popular_hits += 1
    return popular_hits / total if total > 0 else 0.0



# Evaluation Result Container

@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""

    # Ranking metrics (per K value)
    ndcg:       Dict[int, float]
    precision:  Dict[int, float]
    recall:     Dict[int, float]
    hit_rate:   Dict[int, float]
    map_score:  Dict[int, float]
    mrr:        float

    # Beyond-accuracy
    coverage:          float
    diversity:         float
    novelty_score:     float
    popular_bias:      float

    # Meta
    num_users_evaluated:   int
    num_items_total:       int
    model_version:         str
    evaluation_timestamp:  str

    # Promotion decision
    promoted:              bool
    promotion_reasons:     List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "═══ Evaluation Report ═══════════════════════════════",
            f"  Model Version    : {self.model_version}",
            f"  Users Evaluated  : {self.num_users_evaluated:,}",
            f"  Total Products   : {self.num_items_total:,}",
            "",
            "  ── Ranking Metrics ──────────────────────────────",
        ]
        for k in sorted(self.ndcg):
            lines.append(
                f"  @{k:2d}  NDCG={self.ndcg[k]:.4f}  "
                f"P={self.precision[k]:.4f}  "
                f"R={self.recall[k]:.4f}  "
                f"HR={self.hit_rate[k]:.4f}  "
                f"MAP={self.map_score[k]:.4f}"
            )
        lines += [
            f"       MRR  = {self.mrr:.4f}",
            "",
            "  ── Beyond-Accuracy Metrics ──────────────────────",
            f"  Coverage    = {self.coverage:.4f}",
            f"  Diversity   = {self.diversity:.4f}",
            f"  Novelty     = {self.novelty_score:.4f}",
            f"  Pop. Bias   = {self.popular_bias:.4f}",
            "",
            f"  ── Promotion Decision: {' PROMOTED' if self.promoted else ' BLOCKED'} ──",
        ]
        if not self.promoted:
            for reason in self.promotion_reasons:
                lines.append(f"    ✗ {reason}")
        lines.append("═════════════════════════════════════════════════════")
        return "\n".join(lines)



# Main Evaluator

class ModelEvaluator:
    """
    Offline evaluation engine for the TheLook recommendation system.

    Usage:
        evaluator = ModelEvaluator(config)
        result    = evaluator.evaluate(
            test_df         = test_split_df,
            recommendations = {user_id: [product_id, ...]},
            item_categories = {product_id: "category"},
            item_popularity = {product_id: purchase_count},
        )
        print(result.summary())
    """

    def __init__(self, config) -> None:
        self.config = config

    # Public API
   

    def evaluate(
        self,
        test_df:         pd.DataFrame,
        recommendations: Dict[int, List[int]],
        item_categories: Dict[int, str],
        item_popularity: Dict[int, int],
        model_version:   str = "unknown",
    ) -> EvaluationResult:
        """
        Run full offline evaluation.

        Args:
            test_df:         Test split with columns [user_id, product_id, label].
            recommendations: {user_id: [product_id, ...]} sorted by score DESC.
            item_categories: {product_id: category_string}
            item_popularity: {product_id: purchase_count}
            model_version:   String version tag for the result artifact.

        Returns:
            EvaluationResult with all metrics and promotion decision.
        """
        k_values = self.config.k_values

        # Build ground truth: relevant items per user (purchased = 1)
        ground_truth = (
            test_df[test_df["label"] == 1]
            .groupby("user_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        # Filter to users that are in both ground_truth and recommendations
        eval_users = set(ground_truth.keys()) & set(recommendations.keys())
        logger.info("Evaluating %d users …", len(eval_users))

        # ── Per-user metrics 
        ndcg_scores   = {k: [] for k in k_values}
        prec_scores   = {k: [] for k in k_values}
        rec_scores    = {k: [] for k in k_values}
        hr_scores     = {k: [] for k in k_values}
        map_scores    = {k: [] for k in k_values}
        mrr_scores    = []

        for user_id in eval_users:
            recs     = recommendations[user_id]
            relevant = ground_truth[user_id]

            mrr_scores.append(mean_reciprocal_rank(recs, relevant))

            for k in k_values:
                ndcg_scores[k].append(ndcg_at_k(recs, relevant, k))
                prec_scores[k].append(precision_at_k(recs, relevant, k))
                rec_scores[k].append(recall_at_k(recs, relevant, k))
                hr_scores[k].append(hit_rate_at_k(recs, relevant, k))
                map_scores[k].append(average_precision_at_k(recs, relevant, k))

        # ── Aggregate 
        ndcg_agg  = {k: float(np.mean(v)) for k, v in ndcg_scores.items()}
        prec_agg  = {k: float(np.mean(v)) for k, v in prec_scores.items()}
        rec_agg   = {k: float(np.mean(v)) for k, v in rec_scores.items()}
        hr_agg    = {k: float(np.mean(v)) for k, v in hr_scores.items()}
        map_agg   = {k: float(np.mean(v)) for k, v in map_scores.items()}
        mrr_agg   = float(np.mean(mrr_scores))

        # ── Beyond-accuracy metrics 
        max_k      = max(k_values)
        total_items = test_df["product_id"].nunique()

        coverage   = catalogue_coverage(recommendations, total_items)
        diversity  = intra_list_diversity(recommendations, item_categories, k=max_k)
        novelty_sc = novelty(recommendations, item_popularity, k=max_k)
        pop_bias   = popular_item_bias(recommendations, item_popularity, k=max_k)

        # ── Promotion decision 
        promoted, reasons = self._promotion_decision(
            ndcg_at_10   = ndcg_agg.get(10, 0.0),
            prec_at_10   = prec_agg.get(10, 0.0),
            recall_at_10 = rec_agg.get(10, 0.0),
            map_at_10    = map_agg.get(10, 0.0),
            coverage     = coverage,
        )

        from datetime import datetime, timezone
        result = EvaluationResult(
            ndcg               = ndcg_agg,
            precision          = prec_agg,
            recall             = rec_agg,
            hit_rate           = hr_agg,
            map_score          = map_agg,
            mrr                = mrr_agg,
            coverage           = coverage,
            diversity          = diversity,
            novelty_score      = novelty_sc,
            popular_bias       = pop_bias,
            num_users_evaluated= len(eval_users),
            num_items_total    = total_items,
            model_version      = model_version,
            evaluation_timestamp = datetime.now(timezone.utc).isoformat(),
            promoted           = promoted,
            promotion_reasons  = reasons,
        )

        logger.info("\n%s", result.summary())
        return result

    def evaluate_cold_start(
        self,
        test_df:         pd.DataFrame,
        recommendations: Dict[int, List[int]],
        cold_start_users: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate metric quality specifically for cold-start users
        (users with ≤2 historical purchases at training time).

        Returns subset metrics dict for cold-start segment.
        """
        cs_set = set(cold_start_users)
        cs_recs = {u: r for u, r in recommendations.items() if u in cs_set}
        cs_test = test_df[test_df["user_id"].isin(cs_set)]

        ground_truth = (
            cs_test[cs_test["label"] == 1]
            .groupby("user_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        ndcg_vals = []
        hr_vals   = []
        for user_id, recs in cs_recs.items():
            relevant = ground_truth.get(user_id, set())
            ndcg_vals.append(ndcg_at_k(recs, relevant, k=10))
            hr_vals.append(hit_rate_at_k(recs, relevant, k=10))

        result = {
            "cold_start_ndcg_at_10":    float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
            "cold_start_hit_rate_at_10":float(np.mean(hr_vals))   if hr_vals   else 0.0,
            "cold_start_users":         len(cs_recs),
        }
        logger.info("Cold-Start Evaluation: %s", result)
        return result

    def evaluate_segment(
        self,
        test_df:         pd.DataFrame,
        recommendations: Dict[int, List[int]],
        segment_col:     str,
        k:               int = 10,
    ) -> pd.DataFrame:
        """
        Compute NDCG@K broken down by a user attribute (gender, country, age_group).
        Used for fairness analysis.

        Returns:
            DataFrame with columns [segment_value, ndcg_at_k, num_users]
        """
        ground_truth = (
            test_df[test_df["label"] == 1]
            .groupby("user_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        if segment_col not in test_df.columns:
            logger.warning("Segment column '%s' not in test_df.", segment_col)
            return pd.DataFrame()

        user_segment = test_df[["user_id", segment_col]].drop_duplicates("user_id")
        rows = []
        for segment_val, group in user_segment.groupby(segment_col):
            user_ids = group["user_id"].tolist()
            ndcg_vals = [
                ndcg_at_k(recommendations.get(u, []), ground_truth.get(u, set()), k)
                for u in user_ids
                if u in recommendations
            ]
            rows.append({
                "segment_value": segment_val,
                f"ndcg_at_{k}":  float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
                "num_users":     len(ndcg_vals),
            })

        return pd.DataFrame(rows).sort_values(f"ndcg_at_{k}", ascending=False)

    
    # Private
    

    def _promotion_decision(
        self,
        ndcg_at_10:   float,
        prec_at_10:   float,
        recall_at_10: float,
        map_at_10:    float,
        coverage:     float,
    ) -> Tuple[bool, List[str]]:
        """
        Gate model promotion against config thresholds.

        Returns:
            (promoted: bool, failed_reasons: List[str])
        """
        cfg     = self.config
        reasons = []

        checks = [
            (ndcg_at_10,   cfg.min_ndcg_at_10,      f"NDCG@10 {ndcg_at_10:.4f} < {cfg.min_ndcg_at_10}"),
            (prec_at_10,   cfg.min_precision_at_10,  f"Precision@10 {prec_at_10:.4f} < {cfg.min_precision_at_10}"),
            (recall_at_10, cfg.min_recall_at_10,     f"Recall@10 {recall_at_10:.4f} < {cfg.min_recall_at_10}"),
            (map_at_10,    cfg.min_map_at_10,         f"MAP@10 {map_at_10:.4f} < {cfg.min_map_at_10}"),
            (coverage,     cfg.min_coverage,          f"Coverage {coverage:.4f} < {cfg.min_coverage}"),
        ]

        for value, threshold, message in checks:
            if value < threshold:
                reasons.append(message)

        return len(reasons) == 0, reasons



# KFP Component


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.1.0",
        "numpy==1.26.0",
        "pyarrow==13.0.0",
        "lightgbm==4.1.0",
        "scikit-learn==1.3.0",
        "kfp==2.4.0",
    ],
)
def model_evaluation_component(
    test_split:           Input[Dataset],
    ranking_model_dir:    Input[Model],
    evaluation_metrics:   Output[Metrics],
    evaluation_report:    Output[Dataset],
    model_version:        str = "v1",
) -> str:
    """
    KFP component: load test split + trained ranker → compute all metrics.

    Args:
        test_split:        Parquet test split from model_training_component.
        ranking_model_dir: Saved LightGBM model directory.
        evaluation_metrics: KFP Metrics artifact.
        evaluation_report: Output JSON report artifact.
        model_version:     Version string for reporting.

    Returns:
        JSON evaluation report string.
        Also writes evaluation_report artifact with full metrics dict.
    """
    import json
    import logging
    import os
    import pickle
    from datetime import datetime, timezone

    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # ── Load test data 
    log.info("Loading test split …")
    test_df = pd.read_parquet(test_split.path)
    log.info("Test set: %d rows", len(test_df))

    # ── Load ranking model 
    model_path = os.path.join(ranking_model_dir.path, "ranking_model.txt")
    booster    = lgb.Booster(model_file=model_path)

    meta_path  = os.path.join(ranking_model_dir.path, "ranking_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    feature_cols = meta["feature_columns"]

    # ── Generate recommendations 
    log.info("Scoring test candidates …")
    test_sorted = test_df.sort_values("user_id").reset_index(drop=True)
    X_test      = test_sorted[[c for c in feature_cols if c in test_sorted.columns]].fillna(0)
    test_sorted["ranking_score"] = booster.predict(
        X_test, num_iteration=booster.best_iteration
    )

    # Build recommendations dict: {user_id: [product_id sorted by score]}
    recommendations = (
        test_sorted
        .sort_values(["user_id", "ranking_score"], ascending=[True, False])
        .groupby("user_id")["product_id"]
        .apply(list)
        .to_dict()
    )

    # ── Ground truth 
    ground_truth = (
        test_df[test_df["label"] == 1]
        .groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    # ── Helper functions (inlined for KFP isolation) 
    def dcg(rels, k):
        rels = np.asarray(rels[:k], dtype=np.float64)
        if not len(rels): return 0.0
        return float(np.sum((2**rels - 1) / np.log2(np.arange(1, len(rels)+1) + 1)))

    def ndcg(recs, rel, k):
        if not rel: return 0.0
        gains = [1.0 if r in rel else 0.0 for r in recs[:k]]
        ideal = dcg(np.ones(min(len(rel), k)), k)
        return dcg(gains, k) / ideal if ideal > 0 else 0.0

    def prec(recs, rel, k):
        return sum(1 for r in recs[:k] if r in rel) / k if k else 0.0

    def rec(recs, rel, k):
        if not rel: return 0.0
        return sum(1 for r in recs[:k] if r in rel) / len(rel)

    def hr(recs, rel, k):
        return float(any(r in rel for r in recs[:k]))

    def mrr(recs, rel):
        for i, r in enumerate(recs, 1):
            if r in rel: return 1.0 / i
        return 0.0

    def ap(recs, rel, k):
        if not rel: return 0.0
        hits, score = 0, 0.0
        for i, r in enumerate(recs[:k], 1):
            if r in rel:
                hits += 1
                score += hits / i
        return score / min(len(rel), k)

    # ── Compute metrics 
    K_VALS  = [5, 10, 20]
    eval_users = set(ground_truth) & set(recommendations)
    log.info("Evaluating %d users …", len(eval_users))

    metrics_per_k = {
        k: {"ndcg": [], "precision": [], "recall": [], "hit_rate": [], "map": []}
        for k in K_VALS
    }
    mrr_scores = []

    for uid in eval_users:
        recs = recommendations[uid]
        rel  = ground_truth[uid]
        mrr_scores.append(mrr(recs, rel))
        for k in K_VALS:
            metrics_per_k[k]["ndcg"].append(ndcg(recs, rel, k))
            metrics_per_k[k]["precision"].append(prec(recs, rel, k))
            metrics_per_k[k]["recall"].append(rec(recs, rel, k))
            metrics_per_k[k]["hit_rate"].append(hr(recs, rel, k))
            metrics_per_k[k]["map"].append(ap(recs, rel, k))

    agg = {
        k: {m: float(np.mean(v)) for m, v in per_k.items()}
        for k, per_k in metrics_per_k.items()
    }

    # Coverage
    all_recommended = set()
    for recs in recommendations.values():
        all_recommended.update(recs[:10])
    total_products = test_df["product_id"].nunique()
    coverage = len(all_recommended) / total_products if total_products > 0 else 0.0

    # Diversity
    item_categories = {}
    if "category" in test_df.columns:
        item_categories = test_df.set_index("product_id")["category"].to_dict()
    diversity_vals = []
    for recs in list(recommendations.values())[:500]:   # sample for speed
        cats = [item_categories.get(p, "unk") for p in recs[:10]]
        if cats:
            diversity_vals.append(len(set(cats)) / len(cats))
    diversity = float(np.mean(diversity_vals)) if diversity_vals else 0.0

    # ── Promotion thresholds 
    THRESHOLDS = {
        "ndcg_10":   0.30, "precision_10": 0.15,
        "recall_10": 0.10, "map_10":        0.12,
        "coverage":  0.20,
    }
    promotion_failures = []
    if agg[10]["ndcg"]      < THRESHOLDS["ndcg_10"]:
        promotion_failures.append(f"NDCG@10={agg[10]['ndcg']:.4f} < {THRESHOLDS['ndcg_10']}")
    if agg[10]["precision"] < THRESHOLDS["precision_10"]:
        promotion_failures.append(f"P@10={agg[10]['precision']:.4f} < {THRESHOLDS['precision_10']}")
    if agg[10]["recall"]    < THRESHOLDS["recall_10"]:
        promotion_failures.append(f"R@10={agg[10]['recall']:.4f} < {THRESHOLDS['recall_10']}")
    if agg[10]["map"]       < THRESHOLDS["map_10"]:
        promotion_failures.append(f"MAP@10={agg[10]['map']:.4f} < {THRESHOLDS['map_10']}")
    if coverage             < THRESHOLDS["coverage"]:
        promotion_failures.append(f"Coverage={coverage:.4f} < {THRESHOLDS['coverage']}")

    promoted = len(promotion_failures) == 0

    # ── Log KFP Metrics 
    for k in K_VALS:
        evaluation_metrics.log_metric(f"ndcg_at_{k}",      agg[k]["ndcg"])
        evaluation_metrics.log_metric(f"precision_at_{k}",  agg[k]["precision"])
        evaluation_metrics.log_metric(f"recall_at_{k}",     agg[k]["recall"])
        evaluation_metrics.log_metric(f"hit_rate_at_{k}",   agg[k]["hit_rate"])
        evaluation_metrics.log_metric(f"map_at_{k}",        agg[k]["map"])

    evaluation_metrics.log_metric("mrr",       float(np.mean(mrr_scores)))
    evaluation_metrics.log_metric("coverage",  coverage)
    evaluation_metrics.log_metric("diversity", diversity)
    evaluation_metrics.log_metric("promoted",  int(promoted))

    # ── Build full report 
    report = {
        "model_version":       model_version,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "users_evaluated":     len(eval_users),
        "total_products":      total_products,
        "metrics_by_k":        agg,
        "mrr":                 float(np.mean(mrr_scores)),
        "coverage":            coverage,
        "diversity":           diversity,
        "thresholds":          THRESHOLDS,
        "promotion_failures":  promotion_failures,
        "promoted":            promoted,
        "feature_importance":  meta.get("feature_importance", {}),
    }

    # Save report artifact
    pd.DataFrame([report]).to_parquet(evaluation_report.path, index=False)

    status_str = " PROMOTED" if promoted else f" BLOCKED ({len(promotion_failures)} failures)"
    log.info("\n═══ Evaluation Summary ═══")
    for k in K_VALS:
        log.info("  @%-2d NDCG=%.4f  P=%.4f  R=%.4f  MAP=%.4f",
                 k, agg[k]["ndcg"], agg[k]["precision"],
                 agg[k]["recall"], agg[k]["map"])
    log.info("  MRR=%.4f  Coverage=%.4f  Diversity=%.4f", 
             float(np.mean(mrr_scores)), coverage, diversity)
    log.info("  Promotion: %s", status_str)

    return json.dumps(report, indent=2, default=str)