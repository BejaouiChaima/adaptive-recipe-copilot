
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from content_based2 import load_recipes, build_text

from preference_clustering import (
    load_data,
    train_model,
    score_profile,
    N_LIKES,
    TOP_K,
    SEED,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "hybrid_model.pkl"

CONTENT_WEIGHT = 0.50
COLLAB_WEIGHT = 0.50

# Nombre de recettes utilisées pour représenter
# les préférences initiales du nouvel utilisateur.
N_PROFILE_SEEDS = 10

# Nombre d'utilisateurs similaires à rechercher.
N_SIMILAR_USERS = 50

RANDOM_STATE = SEED


# ============================================================
# 2. NORMALISATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(scores) == 0:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def validate_scores(
    content_scores,
    collaborative_scores,
    n_recipes,
):

    content_scores = np.asarray(
        content_scores,
        dtype=np.float32,
    ).reshape(-1)

    collaborative_scores = np.asarray(
        collaborative_scores,
        dtype=np.float32,
    ).reshape(-1)

    if len(content_scores) != n_recipes:
        raise ValueError(
            f"Content-Based : {len(content_scores)} scores "
            f"pour {n_recipes} recettes."
        )

    if len(collaborative_scores) != n_recipes:
        raise ValueError(
            f"Collaborative : {len(collaborative_scores)} scores "
            f"pour {n_recipes} recettes."
        )

    return content_scores, collaborative_scores


# ============================================================
# 3. ENTRAINEMENT CONTENT-BASED
# ============================================================

def train_content_model(recipes, recipe_ids):

    print("\n--- Préparation du catalogue Content-Based ---")

    recipes_indexed = recipes.set_index("RecipeId")

    missing_ids = [
        recipe_id
        for recipe_id in recipe_ids
        if recipe_id not in recipes_indexed.index
    ]

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} recettes collaboratives "
            "sont absentes de recipes_clean.csv."
        )

    catalog = (
        recipes_indexed
        .loc[recipe_ids]
        .reset_index()
    )

    if catalog["RecipeId"].tolist() != list(recipe_ids):
        raise ValueError(
            "L'ordre des recettes ne correspond pas "
            "au catalogue collaboratif."
        )

    texts = build_text(
        catalog,
        include_category=True,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    recipe_matrix = vectorizer.fit_transform(texts)

    print("Catalogue :", len(catalog))
    print("Caractéristiques :", recipe_matrix.shape[1])
    print("Matrice TF-IDF :", recipe_matrix.shape)

    return catalog, vectorizer, recipe_matrix


# ============================================================
# 4. CALCUL HYBRIDE POUR UN UTILISATEUR AVEC HISTORIQUE
# ============================================================

def calculate_hybrid_scores(
    known_recipe_indices,
    model,
):

    n_recipes = model["recipe_matrix"].shape[0]

    known_recipe_indices = list(dict.fromkeys(
        int(idx)
        for idx in known_recipe_indices
        if 0 <= int(idx) < n_recipes
    ))

    if not known_recipe_indices:
        raise ValueError(
            "Le profil doit contenir au moins une recette."
        )

    # --------------------------------------------------------
    # CONTENT-BASED
    # --------------------------------------------------------

    recipe_matrix = model["recipe_matrix"]

    user_vector = recipe_matrix[
        known_recipe_indices
    ].mean(axis=0)

    content_scores = (
        recipe_matrix
        @ np.asarray(user_vector).reshape(-1)
    )

    content_scores = np.asarray(
        content_scores,
        dtype=np.float32,
    ).reshape(-1)

    content_scores = normalize_scores(
        content_scores
    )

    # --------------------------------------------------------
    # COLLABORATIVE
    # --------------------------------------------------------

    preference_model = model["preference_model"]

    collaborative_scores, cluster_id = score_profile(
        known_recipe_indices,
        preference_model["interactions"],
        preference_model["item_similarity"],
        preference_model["kmeans"],
        preference_model["cluster_popularity"],
        preference_model["user_preference_matrix"],
        top_similar_users=N_SIMILAR_USERS,
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    content_scores, collaborative_scores = validate_scores(
        content_scores,
        collaborative_scores,
        n_recipes,
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    hybrid_scores = (
        CONTENT_WEIGHT * content_scores
        + COLLAB_WEIGHT * collaborative_scores
    )

    hybrid_scores = np.asarray(
        hybrid_scores,
        dtype=np.float32,
    ).reshape(-1)

    hybrid_scores[known_recipe_indices] = -np.inf

    return (
        content_scores,
        collaborative_scores,
        hybrid_scores,
        cluster_id,
    )


# ============================================================
# 5. RECOMMANDATION POUR UN NOUVEL UTILISATEUR
# ============================================================

def recommend_hybrid(
    model,
    favorite_ingredients=None,
    favorite_categories=None,
    avoided_ingredients=None,
    max_time=None,
    top_n=10,
):

    """
    Recommandation hybride pour un nouvel utilisateur.

    Etape 1 : construire un profil initial à partir des préférences.
    Etape 2 : identifier les utilisateurs similaires.
    Etape 3 : récupérer leurs recettes appréciées.
    Etape 4 : éliminer les recettes déjà utilisées comme profil
              et appliquer les filtres demandés.
    Etape 5 : affiner avec le Content-Based Filtering.
    Etape 6 : combiner les deux scores.
    """

    recipes = model["recipes"].copy()

    vectorizer = model["vectorizer"]
    recipe_matrix = model["recipe_matrix"]

    favorite_ingredients = favorite_ingredients or []
    favorite_categories = favorite_categories or []
    avoided_ingredients = avoided_ingredients or []

    favorite_ingredients = [
        str(x).strip()
        for x in favorite_ingredients
        if str(x).strip()
    ]

    favorite_categories = [
        str(x).strip()
        for x in favorite_categories
        if str(x).strip()
    ]

    avoided_ingredients = [
        str(x).strip()
        for x in avoided_ingredients
        if str(x).strip()
    ]

    # --------------------------------------------------------
    # ETAPE 1 : CONSTRUIRE LE PROFIL INITIAL
    # --------------------------------------------------------

    print("\n========== PROFIL UTILISATEUR ==========")

    query_parts = (
        favorite_ingredients
        + favorite_categories
    )

    query_text = " ".join(query_parts).strip()

    if not query_text:
        raise ValueError(
            "Sélectionne au moins un ingrédient "
            "ou une catégorie."
        )

    query_vector = vectorizer.transform([
        query_text
    ])

    # Similarité entre les préférences et les recettes.
    initial_content_scores = (
        recipe_matrix @ query_vector.T
    ).toarray().ravel()

    initial_content_scores = normalize_scores(
        initial_content_scores
    )

    n_recipes = len(recipes)

    if len(initial_content_scores) != n_recipes:
        raise ValueError(
            "Erreur de dimension du profil initial."
        )

    # Sélection des recettes qui représentent le mieux
    # les préférences déclarées.
    # Elles servent uniquement à construire le profil.
    seed_count = min(
        N_PROFILE_SEEDS,
        n_recipes,
    )

    seed_indices = np.argsort(
        -initial_content_scores
    )[:seed_count].tolist()

    print("Ingrédients :", favorite_ingredients)
    print("Catégories :", favorite_categories)
    print("Recettes utilisées pour le profil :", len(seed_indices))

    # --------------------------------------------------------
    # ETAPE 2 : IDENTIFIER LES UTILISATEURS SIMILAIRES
    # --------------------------------------------------------

    print("\n========== UTILISATEURS SIMILAIRES ==========")

    preference_model = model["preference_model"]

    collaborative_scores, cluster_id = score_profile(
        seed_indices,
        preference_model["interactions"],
        preference_model["item_similarity"],
        preference_model["kmeans"],
        preference_model["cluster_popularity"],
        preference_model["user_preference_matrix"],
        top_similar_users=N_SIMILAR_USERS,
    )

    collaborative_scores = normalize_scores(
        collaborative_scores
    )

    print("Cluster attribué :", cluster_id)
    print("Utilisateurs similaires recherchés :", N_SIMILAR_USERS)

    # --------------------------------------------------------
    # ETAPE 3 : RECETTES APPRECIEES PAR LES UTILISATEURS
    # SIMILAIRES
    # --------------------------------------------------------

    print("\n========== RECETTES COLLABORATIVES ==========")

    # Le score collaboratif est déjà calculé à partir
    # des interactions positives des voisins similaires.
    # Il est indépendant du classement final Content-Based.

    n_positive_collaborative = int(
        np.sum(collaborative_scores > 0)
    )

    print(
        "Recettes avec un score collaboratif positif :",
        n_positive_collaborative,
    )

    # --------------------------------------------------------
    # ETAPE 4 : EXCLURE LE PROFIL INITIAL ET APPLIQUER
    # LES FILTRES
    # --------------------------------------------------------

    print("\n========== FILTRES ==========")

    eligible_mask = np.ones(
        n_recipes,
        dtype=bool,
    )

    # Exclure les recettes déjà utilisées comme profil.
    eligible_mask[seed_indices] = False

    # Ingrédients à éviter.
    ingredient_text = (
        recipes["RecipeIngredientParts"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    for ingredient in avoided_ingredients:

        ingredient_lower = ingredient.lower()

        mask = ingredient_text.str.contains(
            ingredient_lower,
            regex=False,
            na=False,
        )

        eligible_mask &= ~mask.to_numpy()

    # Temps maximal.
    if max_time is not None:

        time_values = pd.to_numeric(
            recipes["PrepTimeMinutes"],
            errors="coerce",
        )

        time_mask = (
            time_values.notna()
            & (time_values <= max_time)
        )

        eligible_mask &= time_mask.to_numpy()

    eligible_indices = np.flatnonzero(
        eligible_mask
    )

    if len(eligible_indices) == 0:

        print("Aucune recette ne correspond aux filtres.")

        return pd.DataFrame(columns=[
            "RecipeId",
            "Name",
            "RecipeCategory",
            "RecipeIngredientParts",
            "PrepTimeMinutes",
            "ContentScore",
            "CollaborativeScore",
            "HybridScore",
        ]), cluster_id

    print("Recettes restantes après filtres :", len(eligible_indices))

    # --------------------------------------------------------
    # ETAPE 5 : AFFINER AVEC CONTENT-BASED
    # --------------------------------------------------------

    print("\n========== CONTENT-BASED ==========")

    # Le Content-Based utilise les préférences déclarées.
    # Il ne choisit plus les recettes servant à identifier
    # les utilisateurs similaires après le calcul collaboratif.

    content_scores = initial_content_scores.copy()

    # --------------------------------------------------------
    # ETAPE 6 : COMBINER LES SCORES
    # --------------------------------------------------------

    print("\n========== SCORE HYBRIDE ==========")

    content_scores, collaborative_scores = validate_scores(
        content_scores,
        collaborative_scores,
        n_recipes,
    )

    hybrid_scores = (
        model["content_weight"] * content_scores
        + model["collab_weight"] * collaborative_scores
    )

    hybrid_scores = np.asarray(
        hybrid_scores,
        dtype=np.float32,
    ).reshape(-1)

    recipes["ContentScore"] = content_scores
    recipes["CollaborativeScore"] = collaborative_scores
    recipes["HybridScore"] = hybrid_scores

    # Exclure les recettes non éligibles.
    recipes.loc[
        ~eligible_mask,
        "HybridScore",
    ] = -np.inf

    # --------------------------------------------------------
    # ETAPE 7 : CLASSEMENT FINAL
    # --------------------------------------------------------

    recipes = recipes[
        np.isfinite(recipes["HybridScore"])
    ]

    recipes = recipes.sort_values(
        "HybridScore",
        ascending=False,
    )

    results = recipes.head(top_n).copy()

    columns = [
        "RecipeId",
        "Name",
        "RecipeCategory",
        "RecipeIngredientParts",
        "PrepTimeMinutes",
        "ContentScore",
        "CollaborativeScore",
        "HybridScore",
    ]

    print("Nombre de recommandations :", len(results))

    return results[columns].reset_index(drop=True), cluster_id


# ============================================================
# 6. METRIQUES D'EVALUATION
# ============================================================

def compute_metrics(recommended, relevant, k=TOP_K):

    recommended = recommended[:k]
    relevant = set(relevant)

    hits = [
        int(recipe_id in relevant)
        for recipe_id in recommended
    ]

    n_hits = sum(hits)

    precision = n_hits / k

    recall = (
        n_hits / len(relevant)
        if relevant
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    hit_rate = float(n_hits > 0)

    reciprocal_rank = 0.0

    for rank, is_hit in enumerate(hits, start=1):
        if is_hit:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        is_hit / np.log2(rank + 1)
        for rank, is_hit in enumerate(hits, start=1)
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        f"F1@{k}": f1,
        f"HitRate@{k}": hit_rate,
        f"MRR@{k}": reciprocal_rank,
        f"NDCG@{k}": ndcg,
    }


# ============================================================
# 7. EVALUATION HYBRIDE
# ============================================================

def evaluate_hybrid(test_histories, model):

    rng = np.random.default_rng(RANDOM_STATE)

    recipe_ids = model["recipe_ids"]

    rows = []

    print("\n--- Evaluation du modèle hybride ---")

    for position, (user_id, history) in enumerate(
        test_histories.items(),
        start=1,
    ):

        history = np.asarray(
            list(set(history)),
            dtype=int,
        )

        if len(history) <= N_LIKES:
            continue

        shuffled = rng.permutation(history)

        known = shuffled[:N_LIKES].tolist()
        relevant = shuffled[N_LIKES:].tolist()

        (
            content_scores,
            collaborative_scores,
            hybrid_scores,
            cluster_id,
        ) = calculate_hybrid_scores(
            known,
            model,
        )

        top_indices = np.argsort(
            -hybrid_scores
        )[:TOP_K]

        recommended_recipe_ids = [
            recipe_ids[idx]
            for idx in top_indices
        ]

        relevant_recipe_ids = [
            recipe_ids[idx]
            for idx in relevant
        ]

        metrics = compute_metrics(
            recommended_recipe_ids,
            relevant_recipe_ids,
        )

        rows.append({
            "AuthorId": user_id,
            "Cluster": cluster_id,
            "KnownLikes": len(known),
            "HiddenLikes": len(relevant),
            **metrics,
        })

        if position % 10 == 0:
            print(
                f"{position}/{len(test_histories)} "
                "utilisateurs évalués"
            )

    if not rows:
        raise ValueError(
            "Aucun utilisateur valide pour l'évaluation."
        )

    details = pd.DataFrame(rows)

    details_path = (
        OUTPUT_DIR / "hybrid_evaluation_details.csv"
    )

    details.to_csv(
        details_path,
        index=False,
    )

    metric_columns = [
        f"Precision@{TOP_K}",
        f"Recall@{TOP_K}",
        f"F1@{TOP_K}",
        f"HitRate@{TOP_K}",
        f"MRR@{TOP_K}",
        f"NDCG@{TOP_K}",
    ]

    summary = pd.DataFrame([
        {
            "Model": "Hybrid",
            "Metric": metric,
            "Mean": details[metric].mean(),
            "Median": details[metric].median(),
            "Std": details[metric].std(),
            "UsersEvaluated": len(details),
        }
        for metric in metric_columns
    ])

    summary_path = (
        OUTPUT_DIR / "hybrid_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nRésultats moyens :")

    print(
        summary[
            ["Metric", "Mean", "Median", "Std"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nDétails :", details_path)
    print("Résumé :", summary_path)

    return summary, details


# ============================================================
# 8. SAUVEGARDE
# ============================================================

def save_hybrid_model(model):

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nModèle hybride sauvegardé.")
    print("Chemin :", MODEL_PATH)


# ============================================================
# 9. CHARGEMENT
# ============================================================

def load_hybrid_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Lance d'abord hybrid_recommendation.py."
        )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# 10. EXECUTION PRINCIPALE
# ============================================================

def main():

    np.random.seed(RANDOM_STATE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # A. CHARGER LES DONNEES
    # --------------------------------------------------------

    print("\n========== ETAPE 1 : DONNEES ==========")

    interactions, test_histories, recipe_ids = load_data()

    # --------------------------------------------------------
    # B. ENTRAINER LE MODELE COLLABORATIF
    # --------------------------------------------------------

    print("\n========== ETAPE 2 : COLLABORATIVE ==========")

    (
        item_similarity,
        kmeans,
        cluster_popularity,
        user_preference_matrix,
    ) = train_model(interactions)

    preference_model = {
        "item_similarity": item_similarity,
        "kmeans": kmeans,
        "cluster_popularity": cluster_popularity,
        "user_preference_matrix": user_preference_matrix,
        "interactions": interactions,
        "recipe_ids": recipe_ids,
    }

    # --------------------------------------------------------
    # C. ENTRAINER CONTENT-BASED
    # --------------------------------------------------------

    print("\n========== ETAPE 3 : CONTENT-BASED ==========")

    recipes = load_recipes()

    catalog, vectorizer, recipe_matrix = train_content_model(
        recipes,
        recipe_ids,
    )

    recipe_to_idx = {
        recipe_id: idx
        for idx, recipe_id in enumerate(recipe_ids)
    }

    # --------------------------------------------------------
    # D. CONSTRUIRE LE MODELE HYBRIDE
    # --------------------------------------------------------

    print("\n========== ETAPE 4 : HYBRID ==========")

    model = {
        "recipes": catalog,
        "recipe_ids": recipe_ids,
        "recipe_to_idx": recipe_to_idx,
        "vectorizer": vectorizer,
        "recipe_matrix": recipe_matrix,
        "preference_model": preference_model,
        "content_weight": CONTENT_WEIGHT,
        "collab_weight": COLLAB_WEIGHT,
    }

    # --------------------------------------------------------
    # E. EVALUATION
    # --------------------------------------------------------

    print("\n========== ETAPE 5 : EVALUATION ==========")

    evaluate_hybrid(
        test_histories,
        model,
    )

    # --------------------------------------------------------
    # F. SAUVEGARDE
    # --------------------------------------------------------

    print("\n========== ETAPE 6 : SAUVEGARDE ==========")

    save_hybrid_model(model)

    # --------------------------------------------------------
    # G. TEST
    # --------------------------------------------------------

    print("\n========== ETAPE 7 : TEST ==========")

    recommendations, cluster_id = recommend_hybrid(
        model,
        favorite_ingredients=[
            "chicken",
            "tomato",
            "garlic",
        ],
        favorite_categories=[
            "Chicken",
        ],
        avoided_ingredients=[
            "cinnamon",
            "milk",
        ],
        max_time=30,
        top_n=10,
    )

    print("\nCluster :", cluster_id)
    print("\nRecommandations :")

    if recommendations.empty:
        print("Aucune recette trouvée.")
    else:
        print(
            recommendations.to_string(index=False)
        )


if __name__ == "__main__":
    main()
