
from pathlib import Path
import sys
import re
import joblib
import numpy as np
import pandas as pd

from collections import Counter


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

# Chemins nécessaires au chargement des modèles sauvegardés
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "recommendation"))

from src.models.recommendation_models import (
    ContentBasedModel,
    SVDRecommender,
    HybridRecommender
)


from src.recommendation.preference_clustering import recommend_for_user


TRAIN_FILE = ROOT / "data/splits/train_reviews.csv"
TEST_FILE = ROOT / "data/splits/test_reviews.csv"
RECIPES_FILE = ROOT / "data/processed/recipes_clean.csv"

MODEL_PATHS = {
    "Content-Based": ROOT / "models/recommendation/content_model.pkl",
    "SVD": ROOT / "models/recommendation/svd_model.pkl",
    "Hybrid": ROOT / "models/recommendation/hybrid_model.pkl",
    "Preference Clustering": ROOT / "models/preference_model.pkl"
}

OUTPUT_DIR = ROOT / "src/evaluation/results_feedback"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_LIKES = 5
MIN_POSITIVE_RATINGS = 6
MAX_USERS = 500
TOP_K = 10
CANDIDATE_POOL = 100
POSITIVE_THRESHOLD = 4
RANDOM_SEED = 42


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================

def normalize_id(value):
    if pd.isna(value):
        return None

    try:
        number = float(value)
        if number.is_integer():
            return int(number)
    except (ValueError, TypeError):
        pass

    return str(value)


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_ingredients(value):
    if value is None or pd.isna(value):
        return []

    # Food.com peut utiliser c("ingredient") ou ['ingredient']
    ingredients = re.findall(r"""["']([^"']+)["']""", str(value))

    return [
        normalize_text(x)
        for x in ingredients
        if normalize_text(x)
    ]


def extract_recommendations(result):
    """
    Convertit les résultats des modèles en liste :
    [(RecipeId, score), ...]
    """

    recommendations = []

    if result is None:
        return recommendations

    # Résultat DataFrame
    if isinstance(result, pd.DataFrame):

        id_column = next(
            (
                col for col in ["RecipeId", "recipe_id", "recipeId"]
                if col in result.columns
            ),
            None
        )

        if id_column is None:
            raise ValueError(
                "Colonne RecipeId absente des recommandations."
            )

        score_column = next(
            (
                col for col in [
                    "FinalScore",
                    "SimilarityScore",
                    "score",
                    "Score",
                    "predicted_rating"
                ]
                if col in result.columns
            ),
            None
        )

        for _, row in result.iterrows():
            recipe_id = normalize_id(row[id_column])

            if recipe_id is None:
                continue

            score = (
                float(row[score_column])
                if score_column is not None
                and pd.notna(row[score_column])
                else np.nan
            )

            recommendations.append((recipe_id, score))

        return recommendations

    # Résultat dictionnaire
    if isinstance(result, dict):

        if "RecipeId" in result:
            return [
                (
                    normalize_id(result["RecipeId"]),
                    result.get("score", np.nan)
                )
            ]

        raise ValueError("Format dictionnaire non reconnu.")

    # Résultat liste ou tableau
    if isinstance(result, (list, tuple, np.ndarray)):

        for item in result:

            if isinstance(item, dict):
                recipe_id = item.get(
                    "RecipeId",
                    item.get("recipe_id")
                )

                score = item.get(
                    "score",
                    item.get("FinalScore", np.nan)
                )

            elif isinstance(item, (list, tuple, np.ndarray)):
                recipe_id = item[0]
                score = item[1] if len(item) > 1 else np.nan

            else:
                recipe_id = item
                score = np.nan

            recipe_id = normalize_id(recipe_id)

            if recipe_id is not None:
                recommendations.append((recipe_id, score))

        return recommendations

    raise TypeError(
        f"Format de recommandations non reconnu : {type(result)}"
    )


def remove_seen_and_duplicates(recommendations, seen_ids):

    final_results = []
    already_added = set()

    for recipe_id, score in recommendations:

        recipe_id = normalize_id(recipe_id)

        if recipe_id is None:
            continue

        if recipe_id in seen_ids:
            continue

        if recipe_id in already_added:
            continue

        already_added.add(recipe_id)
        final_results.append((recipe_id, score))

        if len(final_results) == CANDIDATE_POOL:
            break

    return final_results


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

print("\nChargement des données...")

train = pd.read_csv(
    TRAIN_FILE,
    usecols=["AuthorId", "RecipeId", "Rating"]
)

test = pd.read_csv(
    TEST_FILE,
    usecols=["AuthorId", "RecipeId", "Rating"]
)

recipes = pd.read_csv(
    RECIPES_FILE,
    usecols=["RecipeId", "RecipeIngredientParts"]
)

for df in [train, test]:
    df["AuthorId"] = df["AuthorId"].apply(normalize_id)
    df["RecipeId"] = df["RecipeId"].apply(normalize_id)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

recipes["RecipeId"] = recipes["RecipeId"].apply(normalize_id)

print("Train :", len(train))
print("Test :", len(test))
print("Recettes :", len(recipes))


# ============================================================
# 4. CONSTRUIRE LES HISTORIQUES ET LE GROUND TRUTH
# ============================================================

train_positive = train[
    train["Rating"] >= POSITIVE_THRESHOLD
].copy()

test_positive = test[
    test["Rating"] >= POSITIVE_THRESHOLD
].copy()

train_by_user = {
    user_id: group
    for user_id, group in train_positive.groupby("AuthorId")
}

test_by_user = {
    user_id: group
    for user_id, group in test_positive.groupby("AuthorId")
}

eligible_users = []

for user_id, train_group in train_by_user.items():

    test_group = test_by_user.get(user_id)

    if test_group is None:
        continue

    train_ids = set(train_group["RecipeId"])
    test_ids = set(test_group["RecipeId"])

    # Il faut au moins 5 recettes positives connues
    # et au moins 1 recette positive à retrouver.
    if len(train_ids) >= N_LIKES and len(test_ids - train_ids) >= 1:
        eligible_users.append(user_id)


rng = np.random.default_rng(RANDOM_SEED)

eligible_users = sorted(eligible_users, key=str)

if not eligible_users:
    raise ValueError(
        "Aucun utilisateur éligible. Vérifie les fichiers train/test."
    )

selected_users = rng.choice(
    eligible_users,
    size=min(MAX_USERS, len(eligible_users)),
    replace=False
)

print("\nUtilisateurs éligibles :", len(eligible_users))
print("Utilisateurs sélectionnés :", len(selected_users))


# Historique connu et ground truth indépendant
user_histories = {}
ground_truth_rows = []

for user_id in selected_users:

    train_group = train_by_user[user_id]

    # Choisir exactement 5 recettes aimées
    chosen = train_group.sample(
        n=N_LIKES,
        random_state=RANDOM_SEED
    )

    liked_ids = chosen["RecipeId"].tolist()
    liked_ratings = chosen["Rating"].astype(float).tolist()

    seen_ids = set(liked_ids)

    # Toutes les recettes positives du test restent cachées.
    test_group = test_by_user[user_id]

    relevant = (
        test_group[
            ~test_group["RecipeId"].isin(seen_ids)
        ]
        .groupby("RecipeId")["Rating"]
        .max()
    )

    if relevant.empty:
        continue

    user_histories[user_id] = {
        "liked_ids": liked_ids,
        "liked_ratings": liked_ratings,
        "seen_ids": seen_ids,
        "relevant": {
            normalize_id(recipe_id): (
                2 if rating >= 5 else 1
            )
            for recipe_id, rating in relevant.items()
        }
    }

    for recipe_id, rating in relevant.items():
        ground_truth_rows.append({
            "user_id": user_id,
            "RecipeId": normalize_id(recipe_id),
            "rating": float(rating),
            "relevance": 2 if rating >= 5 else 1
        })


ground_truth = pd.DataFrame(ground_truth_rows)

ground_truth.to_csv(
    OUTPUT_DIR / "ground_truth.csv",
    index=False
)

print("\nGround truth créé.")
print("Utilisateurs évaluables :", len(user_histories))
print("Recettes pertinentes :", len(ground_truth))


# ============================================================
# 5. CHARGER LES CINQ MODELES
# ============================================================

print("\nChargement des cinq modèles...")

models = {}

for model_name, model_path in MODEL_PATHS.items():

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    print("Chargement :", model_name)

    models[model_name] = joblib.load(model_path)

print("Les cinq modèles sont chargés.")


# ============================================================
# 6. PREPARER LE CATALOGUE POUR CONTENT-BASED 2
# ============================================================

ingredient_map = {}

for _, row in recipes.iterrows():

    recipe_id = row["RecipeId"]

    if recipe_id is None:
        continue

    ingredient_map[recipe_id] = extract_ingredients(
        row["RecipeIngredientParts"]
    )


def build_ingredient_profile(liked_ids):

    counter = Counter()

    for recipe_id in liked_ids:

        ingredients = ingredient_map.get(recipe_id, [])

        # Une occurrence par ingrédient et par recette
        counter.update(set(ingredients))

    # Garder les 12 ingrédients les plus fréquents
    return [
        ingredient
        for ingredient, _ in counter.most_common(12)
    ]


# ============================================================
# 7. FONCTION DE RECOMMANDATION PAR MODELE
# ============================================================

def recommend_content_based(model, liked_ids):

    return model.recommend(
        liked_recipe_ids=liked_ids,
        top_n=CANDIDATE_POOL
    )


def recommend_svd(model, liked_ids, liked_ratings):

    return model.recommend(
        liked_recipe_ids=liked_ids,
        liked_ratings=liked_ratings,
        top_n=CANDIDATE_POOL
    )


def recommend_hybrid(model, liked_ids, liked_ratings):

    return model.recommend(
        liked_recipe_ids=liked_ids,
        liked_ratings=liked_ratings,
        top_n=CANDIDATE_POOL
    )



def recommend_preference_clustering(model, liked_ids):

    # Récupérer les identifiants des recettes depuis le modèle
    recipe_ids = model["recipe_ids"]

    # Construire la correspondance RecipeId -> index
    recipe_to_idx = {
        normalize_id(recipe_id): index
        for index, recipe_id in enumerate(recipe_ids)
    }

    # Convertir les recettes aimées en indices
    liked_indices = []

    for recipe_id in liked_ids:

        recipe_id = normalize_id(recipe_id)
        index = recipe_to_idx.get(recipe_id)

        if index is not None:
            liked_indices.append(index)

    if not liked_indices:
        return []

    # Générer les recommandations avec le modèle existant
    result = recommend_for_user(
        liked_indices=liked_indices,
        disliked_indices=[],
        model_data=model,
        top_n=CANDIDATE_POOL,
        seen_indices=set(liked_indices)
    )

    # La fonction peut retourner (indices, cluster_id)
    if isinstance(result, tuple):
        result = result[0]

    recommendations = []

    for index in result:

        index = int(index)

        if 0 <= index < len(recipe_ids):

            recipe_id = normalize_id(recipe_ids[index])

            recommendations.append(
                (recipe_id, np.nan)
            )

    return recommendations


# ============================================================
# 8. CALCUL DES METRIQUES
# ============================================================

def calculate_metrics(recommendations, relevant, k=10):

    recommended_ids = [
        normalize_id(recipe_id)
        for recipe_id, _ in recommendations[:k]
    ]

    relevant_ids = {
        recipe_id
        for recipe_id, relevance in relevant.items()
        if relevance > 0
    }

    if not relevant_ids:
        return None

    hits = [
        1 if recipe_id in relevant_ids else 0
        for recipe_id in recommended_ids
    ]

    hit_count = sum(hits)

    precision = hit_count / k
    recall = hit_count / len(relevant_ids)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    # AP@K
    cumulative_hits = 0
    precision_sum = 0

    for rank, hit in enumerate(hits, start=1):

        if hit:
            cumulative_hits += 1
            precision_sum += cumulative_hits / rank

    average_precision = (
        precision_sum / min(len(relevant_ids), k)
    )

    # MRR@K
    reciprocal_rank = 0

    for rank, hit in enumerate(hits, start=1):

        if hit:
            reciprocal_rank = 1 / rank
            break

    # NDCG@K
    dcg = 0

    for rank, recipe_id in enumerate(recommended_ids, start=1):

        relevance = relevant.get(recipe_id, 0)

        dcg += (
            (2 ** relevance - 1)
            / np.log2(rank + 1)
        )

    ideal_relevances = sorted(
        relevant.values(),
        reverse=True
    )[:k]

    idcg = sum(
        (2 ** relevance - 1) / np.log2(rank + 1)
        for rank, relevance in enumerate(
            ideal_relevances,
            start=1
        )
    )

    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "F1@10": f1,
        "MAP@10": average_precision,
        "MRR@10": reciprocal_rank,
        "NDCG@10": ndcg,
        "HitRate@10": 1 if hit_count > 0 else 0
    }


# ============================================================
# 9. EXECUTION DES CINQ MODELES
# ============================================================

recommendation_rows = []
per_user_metric_rows = []

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Evaluation :", model_name)
    print("=" * 55)

    for number, (user_id, history) in enumerate(
        user_histories.items(),
        start=1
    ):

        liked_ids = history["liked_ids"]
        liked_ratings = history["liked_ratings"]
        seen_ids = history["seen_ids"]
        relevant = history["relevant"]

        print(
            f"{model_name} - utilisateur "
            f"{number}/{len(user_histories)}",
            end="\r"
        )

        if model_name == "Content-Based":
            raw_result = recommend_content_based(
                model, liked_ids
            )

        
        elif model_name == "SVD":
            raw_result = recommend_svd(
                model, liked_ids, liked_ratings
            )
        elif model_name == "Hybrid":
            raw_result = recommend_hybrid(
                model, liked_ids, liked_ratings
            )



        elif model_name == "Preference Clustering":
            raw_result = recommend_preference_clustering(
                model, liked_ids
            )

        else:
            raise ValueError(
                f"Modèle non reconnu : {model_name}"
            )

        ranked = extract_recommendations(raw_result)

        ranked = remove_seen_and_duplicates(
            ranked,
            seen_ids
        )

        # Garder les 10 premières recommandations
        ranked = ranked[:TOP_K]

        for rank, (recipe_id, score) in enumerate(
            ranked,
            start=1
        ):

            recommendation_rows.append({
                "model": model_name,
                "user_id": user_id,
                "RecipeId": recipe_id,
                "rank": rank,
                "score": score
            })

        scores = calculate_metrics(
            ranked,
            relevant,
            TOP_K
        )

        if scores is not None:

            scores["model"] = model_name
            scores["user_id"] = user_id

            per_user_metric_rows.append(scores)

    print("\nTerminé :", model_name)


# ============================================================
# 10. SAUVEGARDER LES RECOMMANDATIONS
# ============================================================

recommendations_df = pd.DataFrame(
    recommendation_rows
)

recommendations_df.to_csv(
    OUTPUT_DIR / "recommendations.csv",
    index=False
)

print("\nRecommendations sauvegardées.")


# ============================================================
# 11. METRIQUES PAR UTILISATEUR ET PAR MODELE
# ============================================================

per_user_metrics = pd.DataFrame(
    per_user_metric_rows
)

per_user_metrics.to_csv(
    OUTPUT_DIR / "metrics_per_user.csv",
    index=False
)


# ============================================================
# 12. COMPARAISON FINALE DES CINQ MODELES
# ============================================================

metric_columns = [
    "Precision@10",
    "Recall@10",
    "F1@10",
    "MAP@10",
    "MRR@10",
    "NDCG@10",
    "HitRate@10"
]

if per_user_metrics.empty:
    raise ValueError(
        "Aucune métrique calculée. Vérifie les recommandations."
    )

summary = (
    per_user_metrics
    .groupby("model")[metric_columns]
    .mean()
    .reset_index()
)

summary["evaluated_users"] = (
    per_user_metrics
    .groupby("model")["user_id"]
    .nunique()
    .values
)

summary = summary.sort_values(
    "NDCG@10",
    ascending=False
)

summary.to_csv(
    OUTPUT_DIR / "metrics_comparison.csv",
    index=False
)


# ============================================================
# 13. AFFICHAGE FINAL
# ============================================================

print("\n")
print("=" * 75)
print("                 COMPARAISON FINALE DES MODELES")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nFichiers générés dans :", OUTPUT_DIR)
print("- ground_truth.csv")
print("- recommendations.csv")
print("- metrics_per_user.csv")
print("- metrics_comparison.csv")
