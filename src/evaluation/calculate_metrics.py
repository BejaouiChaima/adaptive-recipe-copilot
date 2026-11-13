
from pathlib import Path
import json
import re
import ast
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

QUERIES_PATH = ROOT / "src/evaluation/benchmark_queries.json"
CANDIDATES_PATH = ROOT / "src/evaluation/ground_truth_candidates.csv"
RECOMMENDATIONS_PATH = ROOT / "src/evaluation/recommendations.csv"
GROUND_TRUTH_PATH = ROOT / "src/evaluation/ground_truth.csv"

OUTPUT_PATH = ROOT / "src/evaluation/metrics_results.csv"

K = 10


# ============================================================
# 1. UTILITAIRES
# ============================================================

def parse_ingredients(value):

    if pd.isna(value):
        return []

    value = str(value)

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple)):
            return [str(x).lower().strip() for x in parsed]

    except (ValueError, SyntaxError):
        pass

    return [
        x.lower().strip()
        for x in re.findall(r'"([^"]*)"', value)
    ]


def normalize_id(value):
    return str(value).split(".")[0]


# ============================================================
# 2. METRIQUES DE PERTINENCE
# ============================================================

def precision_at_k(recommended, relevant, k=10):

    top = recommended[:k]

    if not top:
        return 0.0

    hits = sum(recipe in relevant for recipe in top)

    return hits / k


def recall_at_k(recommended, relevant, k=10):

    if not relevant:
        return np.nan

    top = recommended[:k]

    hits = sum(recipe in relevant for recipe in top)

    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k=10):

    return float(
        any(recipe in relevant for recipe in recommended[:k])
    )


def mrr_at_k(recommended, relevant, k=10):

    for rank, recipe in enumerate(recommended[:k], start=1):

        if recipe in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(recommended, relevance_dict, k=10):

    top = recommended[:k]

    dcg = 0.0

    for rank, recipe in enumerate(top, start=1):

        relevance = relevance_dict.get(recipe, 0)

        dcg += (2 ** relevance - 1) / np.log2(rank + 1)

    ideal = sorted(
        relevance_dict.values(),
        reverse=True
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / np.log2(rank + 1)
        for rank, relevance in enumerate(ideal, start=1)
    )

    if idcg == 0:
        return np.nan

    return dcg / idcg


# ============================================================
# 3. INGREDIENT COVERAGE
# ============================================================

def ingredient_coverage(recommended, recipe_data, wanted):

    if not recommended or not wanted:
        return np.nan

    scores = []

    for recipe_id in recommended:

        if recipe_id not in recipe_data:
            continue

        ingredients = recipe_data[recipe_id]

        matched = sum(
            any(
                wanted_ingredient.lower() in ingredient
                for ingredient in ingredients
            )
            for wanted_ingredient in wanted
        )

        scores.append(matched / len(wanted))

    return np.mean(scores) if scores else np.nan


# ============================================================
# 4. CONSTRAINT SATISFACTION
# ============================================================

def constraint_satisfaction(
    recommended,
    recipe_data,
    avoided
):

    if not recommended:
        return np.nan

    if not avoided:
        return 1.0

    scores = []

    for recipe_id in recommended:

        ingredients = recipe_data.get(recipe_id, [])

        valid = not any(
            avoided_ingredient.lower() in ingredient
            for avoided_ingredient in avoided
            for ingredient in ingredients
        )

        scores.append(float(valid))

    return np.mean(scores)


# ============================================================
# 5. DIVERSITY
# ============================================================

def diversity_at_k(recommended, recipe_data):

    if len(recommended) < 2:
        return np.nan

    similarities = []

    for i in range(len(recommended)):

        for j in range(i + 1, len(recommended)):

            ingredients_a = set(
                recipe_data.get(recommended[i], [])
            )

            ingredients_b = set(
                recipe_data.get(recommended[j], [])
            )

            union = ingredients_a | ingredients_b

            if not union:
                continue

            similarity = (
                len(ingredients_a & ingredients_b) / len(union)
            )

            similarities.append(similarity)

    if not similarities:
        return np.nan

    return 1 - np.mean(similarities)


# ============================================================
# 6. CHARGEMENT DES DONNEES
# ============================================================

print("Chargement des questions...")

with open(QUERIES_PATH, encoding="utf-8") as f:
    queries = json.load(f)

print("Chargement des candidats...")

candidates = pd.read_csv(
    CANDIDATES_PATH,
    usecols=["RecipeId", "RecipeIngredientParts"]
)

candidates["RecipeId"] = candidates["RecipeId"].apply(
    normalize_id
)

candidates = candidates.drop_duplicates("RecipeId")

recipe_data = dict(
    zip(
        candidates["RecipeId"],
        candidates["RecipeIngredientParts"].apply(parse_ingredients)
    )
)

print("Chargement des recommandations...")

recommendations = pd.read_csv(RECOMMENDATIONS_PATH)

recommendations["RecipeId"] = recommendations["RecipeId"].apply(
    normalize_id
)

recommendations["query_id"] = recommendations["query_id"].astype(str)

recommendations = recommendations.sort_values(
    ["query_id", "model", "rank"]
)

print("Chargement de la ground truth...")

ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

ground_truth["RecipeId"] = ground_truth["RecipeId"].apply(
    normalize_id
)

ground_truth["query_id"] = ground_truth["query_id"].astype(str)


# ============================================================
# 7. CALCUL
# ============================================================

results = []

for query in queries:

    query_id = query["id"]

    wanted = query.get("wanted_ingredients", [])
    avoided = query.get("avoided_ingredients", [])

    query_gt = ground_truth[
        ground_truth["query_id"] == query_id
    ]

    relevance_dict = dict(
        zip(
            query_gt["RecipeId"],
            query_gt["relevance"]
        )
    )

    relevant = {
        recipe_id
        for recipe_id, relevance in relevance_dict.items()
        if relevance > 0
    }

    query_recommendations = recommendations[
        recommendations["query_id"] == query_id
    ]

    for model in query_recommendations["model"].unique():

        model_results = query_recommendations[
            query_recommendations["model"] == model
        ]

        recommended = (
            model_results["RecipeId"]
            .drop_duplicates()
            .tolist()[:K]
        )

        metrics = {
            "query_id": query_id,
            "model": model,
            "Precision@10": precision_at_k(
                recommended, relevant, K
            ),
            "Recall@10": recall_at_k(
                recommended, relevant, K
            ),
            "NDCG@10": ndcg_at_k(
                recommended, relevance_dict, K
            ),
            "MRR@10": mrr_at_k(
                recommended, relevant, K
            ),
            "HitRate@10": hit_rate_at_k(
                recommended, relevant, K
            ),
            "IngredientCoverage": ingredient_coverage(
                recommended, recipe_data, wanted
            ),
            "ConstraintSatisfaction": constraint_satisfaction(
                recommended, recipe_data, avoided
            ),
            "Diversity@10": diversity_at_k(
                recommended, recipe_data
            )
        }

        results.append(metrics)


# ============================================================
# 8. EXPORT
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nRESULTATS PAR QUESTION ET MODELE")

print(results_df.to_string(index=False))

print("\nMOYENNES PAR MODELE")

summary = (
    results_df
    .groupby("model")
    .mean(numeric_only=True)
    .round(4)
)

print(summary)

summary.to_csv(
    ROOT / "src/evaluation/metrics_summary.csv",
    encoding="utf-8-sig"
)

print("\nFichiers générés :")
print(OUTPUT_PATH)
print(ROOT / "src/evaluation/metrics_summary.csv")
