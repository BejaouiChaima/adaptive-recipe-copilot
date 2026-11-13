
from pathlib import Path
import sys
import pickle
import random
import gc
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ============================================================
# 1. CONFIGURATION ET IMPORTS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# Permet à Python de trouver le package src
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models import recommendation_models as recommendation_models_module

# Compatibilité avec les anciens modèles sauvegardés
# sous le nom "recommendation_models".
sys.modules["recommendation_models"] = recommendation_models_module

from src.models.recommendation_models import (
    ContentBasedModel,
    SVDRecommender,
    HybridRecommender
)
from src.recommendation.content_based2 import prepare_recipes

from src.recommendation.preference_clustering import (
    train_model as train_preference_model,
    recommend_for_user
)


DATA_DIR = ROOT_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models"
RECOMMENDATION_MODEL_DIR = MODEL_DIR / "recommendation"

RESULTS_DIR = ROOT_DIR / "results" / "evaluation"

TRAIN_PATH = SPLIT_DIR / "train_reviews.csv"
TEST_PATH = SPLIT_DIR / "test_reviews.csv"
RECIPES_PATH = DATA_DIR / "processed" / "recipes_clean.csv"

CONTENT_MODEL_PATH = RECOMMENDATION_MODEL_DIR / "content_model.pkl"
SVD_MODEL_PATH = RECOMMENDATION_MODEL_DIR / "svd_model.pkl"
HYBRID_MODEL_PATH = RECOMMENDATION_MODEL_DIR / "hybrid_model.pkl"
CONTENT2_MODEL_PATH = MODEL_DIR / "content_based_eval.pkl"

TOP_K = 10

# Nombre de candidats récupérés avant filtrage.
# On évite de demander les centaines de milliers de recettes
# à chaque appel de recommend().
CANDIDATE_LIMIT = 200

MAX_USERS = 500
RANDOM_STATE = 42
MIN_RATING = 4


# ============================================================
# 2. UTILITAIRES
# ============================================================

def normalize_id(value):
    """Normalise les identifiants de recettes et d'utilisateurs."""

    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


def load_joblib_model(path):
    """Charge un modèle sauvegardé avec joblib."""

    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    print(f"Chargement Joblib : {path.name}")
    return joblib.load(path)


def load_pickle_model(path):
    """Charge un modèle sauvegardé avec pickle."""

    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    print(f"Chargement Pickle : {path.name}")

    with open(path, "rb") as file:
        return pickle.load(file)


def get_recipe_ids_from_results(results):
    """Récupère les identifiants depuis les résultats d'un modèle."""

    if results is None:
        return []

    if isinstance(results, pd.DataFrame):
        if results.empty or "RecipeId" not in results.columns:
            return []

        return results["RecipeId"].map(normalize_id).dropna().tolist()

    if isinstance(results, (list, tuple, np.ndarray)):
        return [normalize_id(x) for x in results if normalize_id(x)]

    return []


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

def load_data():

    print("\nChargement des données...")

    for path in [TRAIN_PATH, TEST_PATH, RECIPES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    recipes = pd.read_csv(
        RECIPES_PATH,
        low_memory=False
    )

    for df in [train, test]:

        required_columns = ["AuthorId", "RecipeId", "Rating"]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Colonnes manquantes dans les données : {missing}"
            )

        df.dropna(
            subset=required_columns,
            inplace=True
        )

        df["AuthorId"] = df["AuthorId"].map(normalize_id)
        df["RecipeId"] = df["RecipeId"].map(normalize_id)

        df["Rating"] = pd.to_numeric(
            df["Rating"],
            errors="coerce"
        )

        df.dropna(
            subset=required_columns,
            inplace=True
        )

        df = df[df["AuthorId"].notna() & df["RecipeId"].notna()]

    if "RecipeId" not in recipes.columns:
        raise ValueError(
            "La colonne RecipeId est absente de recipes_clean.csv."
        )

    recipes["RecipeId"] = recipes["RecipeId"].map(normalize_id)

    print("Train :", len(train))
    print("Test :", len(test))
    print("Recettes :", len(recipes))

    return train, test, recipes


# ============================================================
# 4. CHARGEMENT DES MODÈLES
# ============================================================

def load_models():

    print("\nChargement des modèles...")

    # Ces trois modèles ont été sauvegardés avec Joblib.
    content_model = load_joblib_model(CONTENT_MODEL_PATH)
    svd_model = load_joblib_model(SVD_MODEL_PATH)
    hybrid_model = load_joblib_model(HYBRID_MODEL_PATH)

    # Content-Based 2 a été sauvegardé avec pickle.dump().
    content2_model = load_pickle_model(CONTENT2_MODEL_PATH)

    print("\nVérification des modèles :")

    print("Content-Based :", type(content_model).__name__)
    print("SVD :", type(svd_model).__name__)
    print("Hybrid :", type(hybrid_model).__name__)
    print("Content-Based 2 :", type(content2_model).__name__)

    if not isinstance(content_model, ContentBasedModel):
        raise TypeError("Le modèle Content-Based chargé est invalide.")

    if not isinstance(svd_model, SVDRecommender):
        raise TypeError("Le modèle SVD chargé est invalide.")

    if not isinstance(hybrid_model, HybridRecommender):
        raise TypeError("Le modèle Hybrid chargé est invalide.")

    if not hasattr(content2_model, "recipes"):
        raise TypeError(
            "Le modèle Content-Based 2 ne contient pas l'attribut recipes."
        )

    return (
        content_model,
        svd_model,
        hybrid_model,
        content2_model
    )


# ============================================================
# 5. CONSTRUCTION DE PREFERENCE CLUSTERING
# ============================================================

def build_preference_model(train):

    print("\nConstruction de Preference Clustering...")

    positive_train = train[
        train["Rating"] >= MIN_RATING
    ].copy()

    positive_train = positive_train.drop_duplicates(
        subset=["AuthorId", "RecipeId"]
    )

    user_ids = (
        positive_train["AuthorId"]
        .drop_duplicates()
        .tolist()
    )

    recipe_ids = (
        positive_train["RecipeId"]
        .drop_duplicates()
        .tolist()
    )

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    recipe_to_idx = {
        recipe_id: idx
        for idx, recipe_id in enumerate(recipe_ids)
    }

    rows = positive_train["AuthorId"].map(
        user_to_idx
    ).to_numpy(dtype=np.int32)

    cols = positive_train["RecipeId"].map(
        recipe_to_idx
    ).to_numpy(dtype=np.int32)

    interactions = csr_matrix(
        (
            np.ones(len(positive_train), dtype=np.float32),
            (rows, cols)
        ),
        shape=(len(user_ids), len(recipe_ids)),
        dtype=np.float32
    )

    del positive_train, rows, cols
    gc.collect()

    print("Utilisateurs :", interactions.shape[0])
    print("Recettes :", interactions.shape[1])
    print("Interactions :", interactions.nnz)

    preference_model = train_preference_model(interactions)

    preference_model["recipe_ids"] = recipe_ids
    preference_model["user_ids"] = user_ids
    preference_model["user_to_idx"] = user_to_idx
    preference_model["recipe_to_idx"] = recipe_to_idx

    return preference_model


# ============================================================
# 6. CONSTRUCTION DU CATALOGUE COMMUN
# ============================================================

def build_common_catalog(
    content_model,
    svd_model,
    content2_model,
    preference_model
):

    print("\nConstruction du catalogue commun...")

    content_ids = {
        normalize_id(x)
        for x in content_model.recipe_to_index.keys()
    }

    svd_ids = {
        normalize_id(x)
        for x in svd_model.item_to_index.keys()
    }

    content2_ids = {
        normalize_id(x)
        for x in content2_model.recipes["RecipeId"].dropna()
    }

    preference_ids = {
        normalize_id(x)
        for x in preference_model["recipe_ids"]
    }

    common_ids = (
        content_ids
        & svd_ids
        & content2_ids
        & preference_ids
    )

    common_ids.discard(None)

    print("\nCatalogues disponibles :")
    print("Content-Based :", len(content_ids))
    print("SVD :", len(svd_ids))
    print("Content-Based 2 :", len(content2_ids))
    print("Preference Clustering :", len(preference_ids))

    print("\nCatalogue commun :", len(common_ids))

    if not common_ids:
        raise ValueError(
            "Le catalogue commun est vide. "
            "Vérifiez les identifiants des recettes dans les modèles."
        )

    return common_ids


# ============================================================
# 7. PRÉPARATION DES UTILISATEURS
# ============================================================

def prepare_users(train, test, common_catalog):

    print("\nPréparation des utilisateurs...")

    positive_train = train[
        train["Rating"] >= MIN_RATING
    ].copy()

    positive_test = test[
        test["Rating"] >= MIN_RATING
    ].copy()

    # Les utilisateurs sélectionnés ont exactement cinq recettes
    # positives dans le train.
    train_counts = (
        positive_train.groupby("AuthorId")["RecipeId"]
        .nunique()
    )

    eligible_users = train_counts[
        train_counts == 5
    ].index.tolist()

    # Les recettes cachées doivent appartenir au catalogue commun.
    positive_test = positive_test[
        positive_test["RecipeId"].isin(common_catalog)
    ].copy()

    test_users = set(
        positive_test["AuthorId"].unique()
    )

    eligible_users = [
        user_id
        for user_id in eligible_users
        if user_id in test_users
    ]

    random.seed(RANDOM_STATE)
    random.shuffle(eligible_users)

    selected_users = eligible_users[:MAX_USERS]

    if not selected_users:
        raise ValueError(
            "Aucun utilisateur éligible. "
            "Vérifiez le split et le catalogue commun."
        )

    print("Utilisateurs éligibles :", len(eligible_users))
    print("Utilisateurs sélectionnés :", len(selected_users))

    return (
        positive_train,
        positive_test,
        selected_users
    )


# ============================================================
# 8. MÉTRIQUES
# ============================================================

def calculate_metrics(recommended_ids, relevant_ids):

    recommended_ids = [
        normalize_id(x)
        for x in recommended_ids
        if normalize_id(x) is not None
    ]

    relevant_ids = {
        normalize_id(x)
        for x in relevant_ids
        if normalize_id(x) is not None
    }

    # Éliminer les doublons éventuels sans changer l'ordre.
    recommended_ids = list(dict.fromkeys(recommended_ids))

    relevance = [
        1 if recipe_id in relevant_ids else 0
        for recipe_id in recommended_ids[:TOP_K]
    ]

    hits = sum(relevance)

    precision = hits / TOP_K

    recall = (
        hits / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    dcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(relevance, start=1)
    )

    ideal_relevance = [1] * min(len(relevant_ids), TOP_K)

    idcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevance, start=1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    hit_rate = 1.0 if hits > 0 else 0.0

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg,
        "HitRate@10": hit_rate,
        "Hits": hits,
        "ReturnedRecipes": len(recommended_ids[:TOP_K])
    }


# ============================================================
# 9. PROFIL CONTENT-BASED 2
# ============================================================

def build_ingredient_profile(liked_ids, recipes_df):

    liked_recipes = recipes_df[
        recipes_df["RecipeId"].isin(liked_ids)
    ]

    ingredients = []

    if "IngredientsList" in liked_recipes.columns:

        for recipe_ingredients in liked_recipes["IngredientsList"]:

            if isinstance(recipe_ingredients, (list, tuple, set)):
                ingredients.extend(
                    str(ingredient).strip()
                    for ingredient in recipe_ingredients
                    if ingredient is not None
                )

    ingredient_counts = Counter(
        ingredient
        for ingredient in ingredients
        if ingredient
    )

    wanted_ingredients = [
        ingredient
        for ingredient, count in ingredient_counts.most_common(30)
    ]

    category = None

    if "RecipeCategory" in liked_recipes.columns:

        categories = (
            liked_recipes["RecipeCategory"]
            .dropna()
            .astype(str)
        )

        categories = categories[
            categories.str.strip() != ""
        ]

        if not categories.empty:
            category = categories.mode().iloc[0]

    return wanted_ingredients, category


# ============================================================
# 10. FILTRAGE DES RECOMMANDATIONS
# ============================================================

def filter_recommendations(
    results,
    common_catalog,
    liked_ids
):

    recipe_ids = get_recipe_ids_from_results(results)

    if not recipe_ids:
        return []

    liked_ids = {
        normalize_id(x)
        for x in liked_ids
    }

    filtered = []

    for recipe_id in recipe_ids:

        if recipe_id in common_catalog and recipe_id not in liked_ids:
            if recipe_id not in filtered:
                filtered.append(recipe_id)

        if len(filtered) >= TOP_K:
            break

    return filtered


# ============================================================
# 11. ENREGISTREMENT DES RÉSULTATS
# ============================================================

def evaluate_and_save(
    results,
    user_id,
    model_name,
    recommended_ids,
    relevant_ids,
    liked_ids
):

    metrics = calculate_metrics(
        recommended_ids,
        relevant_ids
    )

    results.append({
        "UserId": user_id,
        "Model": model_name,
        "KnownLikes": len(liked_ids),
        "HiddenRecipes": len(relevant_ids),
        **metrics
    })


# ============================================================
# 12. ÉVALUATION D'UN UTILISATEUR
# ============================================================

def evaluate_one_user(
    user_id,
    positive_train,
    positive_test,
    recipes_df,
    models,
    common_catalog,
    preference_model
):

    content_model, svd_model, hybrid_model, content2_model = models

    results = []

    # --------------------------------------------------------
    # HISTORIQUE CONNU
    # --------------------------------------------------------

    user_train = positive_train[
        positive_train["AuthorId"] == user_id
    ].drop_duplicates("RecipeId")

    liked_ids = user_train["RecipeId"].tolist()
    liked_ratings = user_train["Rating"].tolist()

    # --------------------------------------------------------
    # RECETTES CACHÉES
    # --------------------------------------------------------

    user_test = positive_test[
        positive_test["AuthorId"] == user_id
    ]

    relevant_ids = set(
        user_test["RecipeId"].unique()
    )

    relevant_ids &= common_catalog

    if not relevant_ids:
        return results

    # --------------------------------------------------------
    # 1. CONTENT-BASED
    # --------------------------------------------------------

    try:

        content_results = content_model.recommend(
            liked_ids,
            top_n=CANDIDATE_LIMIT
        )

        recommended = filter_recommendations(
            content_results,
            common_catalog,
            liked_ids
        )

        evaluate_and_save(
            results,
            user_id,
            "Content-Based",
            recommended,
            relevant_ids,
            liked_ids
        )

    except Exception as error:

        print(
            f"\nContent-Based - utilisateur {user_id} : {error}"
        )

    # --------------------------------------------------------
    # 2. SVD
    # --------------------------------------------------------

    try:

        svd_results = svd_model.recommend(
            liked_ids,
            liked_ratings=liked_ratings,
            top_n=CANDIDATE_LIMIT
        )

        recommended = filter_recommendations(
            svd_results,
            common_catalog,
            liked_ids
        )

        evaluate_and_save(
            results,
            user_id,
            "SVD",
            recommended,
            relevant_ids,
            liked_ids
        )

    except Exception as error:

        print(
            f"\nSVD - utilisateur {user_id} : {error}"
        )

    # --------------------------------------------------------
    # 3. HYBRID
    # --------------------------------------------------------

    try:

        old_candidate_pool = hybrid_model.candidate_pool

        try:

            hybrid_model.candidate_pool = CANDIDATE_LIMIT

            hybrid_results = hybrid_model.recommend(
                liked_ids,
                liked_ratings=liked_ratings,
                top_n=CANDIDATE_LIMIT
            )

        finally:

            hybrid_model.candidate_pool = old_candidate_pool

        recommended = filter_recommendations(
            hybrid_results,
            common_catalog,
            liked_ids
        )

        evaluate_and_save(
            results,
            user_id,
            "Hybrid",
            recommended,
            relevant_ids,
            liked_ids
        )

    except Exception as error:

        print(
            f"\nHybrid - utilisateur {user_id} : {error}"
        )

    # --------------------------------------------------------
    # 4. CONTENT-BASED 2
    # --------------------------------------------------------

    try:

        wanted_ingredients, category = build_ingredient_profile(
            liked_ids,
            recipes_df
        )

        if not wanted_ingredients and not category:
            raise ValueError(
                "Aucun ingrédient ou catégorie disponible."
            )

        content2_results = content2_model.recommend(
            wanted_ingredients=wanted_ingredients,
            category=category,
            top_n=CANDIDATE_LIMIT
        )

        recommended = filter_recommendations(
            content2_results,
            common_catalog,
            liked_ids
        )

        evaluate_and_save(
            results,
            user_id,
            "Content-Based 2",
            recommended,
            relevant_ids,
            liked_ids
        )

    except Exception as error:

        print(
            f"\nContent-Based 2 - utilisateur {user_id} : {error}"
        )

    # --------------------------------------------------------
    # 5. PREFERENCE CLUSTERING
    # --------------------------------------------------------

    try:

        recipe_to_idx = preference_model["recipe_to_idx"]

        liked_indices = [
            recipe_to_idx[recipe_id]
            for recipe_id in liked_ids
            if recipe_id in recipe_to_idx
        ]

        seen_indices = set(liked_indices)

        preference_indices, cluster_id = recommend_for_user(
            liked_indices=liked_indices,
            disliked_indices=[],
            model_data=preference_model,
            top_n=CANDIDATE_LIMIT,
            seen_indices=seen_indices
        )

        preference_recipe_ids = preference_model["recipe_ids"]

        preference_recommended = [
            normalize_id(preference_recipe_ids[idx])
            for idx in preference_indices
            if 0 <= idx < len(preference_recipe_ids)
        ]

        preference_recommended = [
            recipe_id
            for recipe_id in preference_recommended
            if recipe_id in common_catalog
            and recipe_id not in liked_ids
        ]

        preference_recommended = list(
            dict.fromkeys(preference_recommended)
        )[:TOP_K]

        metrics = calculate_metrics(
            preference_recommended,
            relevant_ids
        )

        results.append({
            "UserId": user_id,
            "Model": "Preference Clustering",
            "KnownLikes": len(liked_ids),
            "HiddenRecipes": len(relevant_ids),
            "ClusterID": cluster_id,
            **metrics
        })

    except Exception as error:

        print(
            f"\nPreference Clustering - utilisateur {user_id} : {error}"
        )

    return results


# ============================================================
# 13. PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("COMPARAISON DES CINQ MODÈLES DE RECOMMANDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CHARGEMENT DES DONNÉES
    # --------------------------------------------------------

    train, test, recipes = load_data()

    # --------------------------------------------------------
    # CHARGEMENT DES MODÈLES
    # --------------------------------------------------------

    models = load_models()

    (
        content_model,
        svd_model,
        hybrid_model,
        content2_model
    ) = models

    # --------------------------------------------------------
    # PRÉPARATION CONTENT-BASED 2
    # --------------------------------------------------------

    print("\nPréparation des recettes Content-Based 2...")

    recipes_df = prepare_recipes(recipes)

    recipes_df["RecipeId"] = (
        recipes_df["RecipeId"].map(normalize_id)
    )

    # --------------------------------------------------------
    # CONSTRUCTION PREFERENCE CLUSTERING
    # --------------------------------------------------------

    preference_model = build_preference_model(train)

    # --------------------------------------------------------
    # CATALOGUE COMMUN
    # --------------------------------------------------------

    common_catalog = build_common_catalog(
        content_model,
        svd_model,
        content2_model,
        preference_model
    )

    # --------------------------------------------------------
    # PRÉPARATION DES UTILISATEURS
    # --------------------------------------------------------

    (
        positive_train,
        positive_test,
        selected_users
    ) = prepare_users(
        train,
        test,
        common_catalog
    )

    # --------------------------------------------------------
    # ÉVALUATION
    # --------------------------------------------------------

    print("\nDébut de l'évaluation...\n")

    all_results = []

    for number, user_id in enumerate(
        selected_users,
        start=1
    ):

        user_results = evaluate_one_user(
            user_id=user_id,
            positive_train=positive_train,
            positive_test=positive_test,
            recipes_df=recipes_df,
            models=models,
            common_catalog=common_catalog,
            preference_model=preference_model
        )

        all_results.extend(user_results)

        if number % 10 == 0 or number == len(selected_users):

            print(
                f"{number}/{len(selected_users)} utilisateurs traités "
                f"({len(all_results)} résultats)"
            )

    if not all_results:
        raise ValueError(
            "Aucun résultat obtenu pendant l'évaluation."
        )

    # --------------------------------------------------------
    # SAUVEGARDE DES RÉSULTATS
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    detailed_df = pd.DataFrame(all_results)

    detailed_path = (
        RESULTS_DIR / "five_models_detailed_results.csv"
    )

    detailed_df.to_csv(
        detailed_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # COMPARAISON DES MÉTRIQUES
    # --------------------------------------------------------

    metrics_columns = [
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "HitRate@10"
    ]

    comparison = (
        detailed_df.groupby("Model")[metrics_columns]
        .mean()
        .reset_index()
    )

    comparison = comparison.sort_values(
        "NDCG@10",
        ascending=False
    ).reset_index(drop=True)

    comparison_path = (
        RESULTS_DIR / "five_models_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # CLASSEMENT
    # --------------------------------------------------------

    ranking = comparison.copy()

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    ranking_path = (
        RESULTS_DIR / "five_models_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # NOMBRE D'UTILISATEURS ÉVALUÉS PAR MODÈLE
    # --------------------------------------------------------

    users_per_model = (
        detailed_df.groupby("Model")["UserId"]
        .nunique()
        .reset_index(name="NumberOfUsers")
    )

    print("\nNombre d'utilisateurs évalués par modèle :")
    print(users_per_model.to_string(index=False))

    # --------------------------------------------------------
    # AFFICHAGE FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉSULTATS DE LA COMPARAISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nClassement basé sur NDCG@10 :")

    for _, row in ranking.iterrows():

        print(
            f"{int(row['Rank'])}. {row['Model']} "
            f"- NDCG@10 : {row['NDCG@10']:.4f}"
        )

    print("\nFichiers sauvegardés :")
    print(detailed_path)
    print(comparison_path)
    print(ranking_path)

    print("\nÉvaluation terminée.")


if __name__ == "__main__":
    main()
