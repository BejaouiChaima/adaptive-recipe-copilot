from pathlib import Path
import json
import re
import joblib
import numpy as np
import pandas as pd
import sys

# ============================================================
# 1. CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
QUERY_FILE = ROOT / "src/evaluation/benchmark_queries.json"
CANDIDATES_FILE = ROOT / "src/evaluation/ground_truth_candidates.csv"
MODEL_FILE = ROOT / "models/content_based_model.pkl"

OUTPUT_DIR = ROOT / "src/evaluation/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.csv"
RECOMMENDATIONS_FILE = OUTPUT_DIR / "recommendations.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.csv"

TOP_K = 10


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def to_number(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float, np.number)):
        return float(value)

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def time_to_minutes(value):
    """
    Accepte les formats Food.com : PT35M, PT1H20M,
    ainsi que des valeurs comme 45 min ou 1 hour.
    """
    if value is None or pd.isna(value):
        return None

    value = str(value).strip().upper()

    iso = re.match(
        r"PT(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?",
        value
    )

    if iso and (iso.group(1) or iso.group(2)):
        hours = float(iso.group(1) or 0)
        minutes = float(iso.group(2) or 0)
        return hours * 60 + minutes

    number = to_number(value)

    if number is None:
        return None

    if "HOUR" in value or "HR" in value:
        return number * 60

    return number


def extract_ingredients(value):
    """
    Extrait les ingrédients depuis le format :
    c("ingredient 1", "ingredient 2")
    """
    if value is None or pd.isna(value):
        return []

    return [
        normalize_text(x)
        for x in re.findall(r'"([^"]+)"', str(value))
    ]


def as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [x.strip() for x in re.split(r"[,;|]", value) if x.strip()]

    return []


# ============================================================
# 3. CHARGER LES 30 QUESTIONS
# ============================================================

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    query_data = json.load(f)

if isinstance(query_data, list):
    queries = query_data
elif isinstance(query_data, dict):
    if isinstance(query_data.get("queries"), list):
        queries = query_data["queries"]
    else:
        queries = []
        for query_id, values in query_data.items():
            if isinstance(values, dict):
                item = values.copy()
                item.setdefault("query_id", query_id)
                queries.append(item)
else:
    raise ValueError("Format de benchmark_queries.json non reconnu.")

if not queries:
    raise ValueError("Aucune question trouvée dans benchmark_queries.json.")

print("Nombre de questions :", len(queries))


# ============================================================
# 4. CHARGER LES CANDIDATES ET CREER LE GROUND TRUTH
# ============================================================

candidates = pd.read_csv(CANDIDATES_FILE)

required_columns = {
    "query_id",
    "RecipeId",
    "ingredient_coverage",
    "RecipeIngredientParts"
}

missing = required_columns - set(candidates.columns)

if missing:
    raise ValueError(f"Colonnes manquantes dans candidates CSV : {missing}")


ground_truth_rows = []

for query in queries:

    query_id = str(query.get("query_id", query.get("id", "")))

    if not query_id:
        continue

    wanted = as_list(query.get("wanted_ingredients", []))
    avoided = as_list(query.get("avoided_ingredients", []))

    max_calories = to_number(query.get("max_calories"))
    max_time = time_to_minutes(query.get("max_time"))

    query_candidates = candidates[
        candidates["query_id"].astype(str) == query_id
    ].copy()

    for _, recipe in query_candidates.iterrows():

        coverage = to_number(recipe.get("ingredient_coverage"))

        if coverage is None:
            matched = to_number(recipe.get("matched_count")) or 0
            total = to_number(recipe.get("wanted_count")) or len(wanted)
            coverage = matched / total if total else 0

        # Certains fichiers peuvent enregistrer la couverture en pourcentage.
        if coverage > 1:
            coverage = coverage / 100

        # Vérifier les ingrédients à éviter.
        recipe_ingredients = extract_ingredients(
            recipe.get("RecipeIngredientParts")
        )

        avoided_found = any(
            normalize_text(avoid) in ingredient
            for avoid in avoided
            for ingredient in recipe_ingredients
        )

        if avoided_found:
            continue

        # Vérifier la limite calorique.
        calories = to_number(recipe.get("Calories"))

        if (
            max_calories is not None
            and calories is not None
            and calories > max_calories
        ):
            continue

        # Vérifier la limite de temps.
        recipe_time = time_to_minutes(recipe.get("TotalTime"))

        if (
            max_time is not None
            and recipe_time is not None
            and recipe_time > max_time
        ):
            continue

        # Pertinence graduée selon la couverture des ingrédients.
        if coverage >= 0.99:
            relevance = 2
        elif coverage >= 0.50:
            relevance = 1
        else:
            relevance = 0

        if relevance > 0:
            ground_truth_rows.append({
                "query_id": query_id,
                "RecipeId": recipe["RecipeId"],
                "relevance": relevance
            })


ground_truth = pd.DataFrame(
    ground_truth_rows,
    columns=["query_id", "RecipeId", "relevance"]
)

ground_truth = ground_truth.drop_duplicates(
    subset=["query_id", "RecipeId"]
)

ground_truth.to_csv(GROUND_TRUTH_FILE, index=False)

print("\nGround truth créé :", GROUND_TRUTH_FILE)
print("Nombre de recettes pertinentes :", len(ground_truth))
print(
    "Questions avec au moins une recette pertinente :",
    ground_truth["query_id"].nunique()
)


# ============================================================
# 5. CHARGER LE MODELE SANS LE REENTRAINER
# ============================================================

model = joblib.load(MODEL_FILE)

if not hasattr(model, "recommend"):
    raise TypeError(
        "Le fichier chargé ne contient pas un modèle avec recommend()."
    )

print("\nModèle chargé :", MODEL_FILE)


# ============================================================
# 6. EXECUTER LE MODELE SUR LES 30 QUESTIONS
# ============================================================

recommendation_rows = []

for index, query in enumerate(queries, start=1):

    query_id = str(query.get("query_id", query.get("id", "")))

    print(f"Question {index}/{len(queries)} : {query_id}")

    results = model.recommend(
        wanted_ingredients=query.get("wanted_ingredients"),
        avoided_ingredients=query.get("avoided_ingredients"),
        max_calories=query.get("max_calories"),
        max_time=query.get("max_time"),
        category=query.get("category"),
        top_n=TOP_K
    )

    if results is None or len(results) == 0:
        continue

    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    if "RecipeId" not in results.columns:
        raise ValueError(
            f"RecipeId absent des résultats pour {query_id}"
        )

    score_column = next(
        (
            col for col in ["FinalScore", "SimilarityScore"]
            if col in results.columns
        ),
        None
    )

    for rank, (_, recipe) in enumerate(
        results.head(TOP_K).iterrows(),
        start=1
    ):
        recommendation_rows.append({
            "query_id": query_id,
            "RecipeId": recipe["RecipeId"],
            "rank": rank,
            "score": (
                recipe[score_column]
                if score_column else np.nan
            )
        })


recommendations = pd.DataFrame(
    recommendation_rows,
    columns=["query_id", "RecipeId", "rank", "score"]
)

recommendations.to_csv(RECOMMENDATIONS_FILE, index=False)

print("\nRecommendations créées :", RECOMMENDATIONS_FILE)
print("Nombre de recommandations :", len(recommendations))


# ============================================================
# 7. CALCUL DES METRIQUES
# ============================================================

def calculate_metrics(recommended_ids, relevant_items, k=10):

    recommended_ids = list(recommended_ids[:k])

    relevant_items = {
        str(recipe_id): int(rel)
        for recipe_id, rel in relevant_items.items()
    }

    relevant_ids = {
        recipe_id
        for recipe_id, relevance in relevant_items.items()
        if relevance > 0
    }

    if not relevant_ids:
        return None

    recommended_str = [str(x) for x in recommended_ids]

    hits = [
        1 if recipe_id in relevant_ids else 0
        for recipe_id in recommended_str
    ]

    hit_count = sum(hits)

    precision = hit_count / k
    recall = hit_count / len(relevant_ids)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    # Average Precision @ K
    precisions_at_hits = []
    cumulative_hits = 0

    for rank, hit in enumerate(hits, start=1):
        if hit:
            cumulative_hits += 1
            precisions_at_hits.append(cumulative_hits / rank)

    average_precision = (
        sum(precisions_at_hits) / min(len(relevant_ids), k)
        if relevant_ids else 0
    )

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank, hit in enumerate(hits, start=1):
        if hit:
            reciprocal_rank = 1 / rank
            break

    # NDCG @ K avec pertinence graduée
    dcg = 0

    for rank, recipe_id in enumerate(recommended_str, start=1):
        relevance = relevant_items.get(recipe_id, 0)
        dcg += (2 ** relevance - 1) / np.log2(rank + 1)

    ideal_relevances = sorted(
        relevant_items.values(),
        reverse=True
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / np.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"F1@{k}": f1,
        f"MAP@{k}": average_precision,
        f"MRR@{k}": reciprocal_rank,
        f"NDCG@{k}": ndcg,
        f"HitRate@{k}": 1 if hit_count > 0 else 0
    }


# Dictionnaire du ground truth par question.
ground_truth_by_query = {
    query_id: group.set_index("RecipeId")["relevance"].to_dict()
    for query_id, group in ground_truth.groupby("query_id")
}

metric_rows = []

for query_id, group in recommendations.groupby("query_id"):

    relevant_items = ground_truth_by_query.get(query_id, {})

    if not relevant_items:
        continue

    scores = calculate_metrics(
        recommended_ids=group.sort_values("rank")["RecipeId"],
        relevant_items=relevant_items,
        k=TOP_K
    )

    if scores is not None:
        scores["query_id"] = query_id
        metric_rows.append(scores)


per_query_metrics = pd.DataFrame(metric_rows)

if per_query_metrics.empty:
    raise ValueError(
        "Aucune métrique calculée. Vérifie le chevauchement "
        "entre recommendations.csv et ground_truth.csv."
    )

metric_columns = [
    col for col in per_query_metrics.columns
    if col != "query_id"
]

final_metrics = (
    per_query_metrics[metric_columns]
    .mean()
    .to_frame(name="value")
    .reset_index()
    .rename(columns={"index": "metric"})
)

final_metrics.to_csv(METRICS_FILE, index=False)

per_query_metrics.to_csv(
    OUTPUT_DIR / "metrics_per_query.csv",
    index=False
)


# ============================================================
# 8. AFFICHER LES RESULTATS FINAUX
# ============================================================

print("\n" + "=" * 55)
print("              METRIQUES FINALES")
print("=" * 55)

print(final_metrics.to_string(index=False))

print("\nNombre de questions évaluées :", len(per_query_metrics))
print("\nFichiers générés :")
print(GROUND_TRUTH_FILE)
print(RECOMMENDATIONS_FILE)
print(METRICS_FILE)
print(OUTPUT_DIR / "metrics_per_query.csv")