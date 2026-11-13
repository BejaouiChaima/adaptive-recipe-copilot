
import sys
import random
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# CHEMINS DU PROJET
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

# Les scripts et les modèles sont dans src/models
MODELS_DIR = ROOT_DIR / "src" / "models"

# Permet de retrouver les classes lors du chargement des PKL
sys.path.insert(0, str(MODELS_DIR))

DATA_DIR = ROOT_DIR / "data" / "splits"

TRAIN_PATH = DATA_DIR / "train_reviews.csv"
TEST_PATH = DATA_DIR / "test_reviews.csv"

CONTENT_MODEL_PATH = MODELS_DIR / "content_model.pkl"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"

RESULTS_PATH = ROOT_DIR / "tests" / "recommendation_models_test_results.csv"


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

N_USERS = 500
TOP_K = 10
RANDOM_SEED = 42


# --------------------------------------------------
# CHARGEMENT DES DONNEES
# --------------------------------------------------

print("Chargement des données...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for df in (train, test):
    df["AuthorId"] = df["AuthorId"].astype(str)
    df["RecipeId"] = df["RecipeId"].astype(str)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

train = train.dropna(
    subset=["AuthorId", "RecipeId", "Rating"]
)

test = test.dropna(
    subset=["AuthorId", "RecipeId", "Rating"]
)

train_positive = train[train["Rating"] >= 4].copy()
test_positive = test[test["Rating"] >= 4].copy()

print("Données chargées avec succès.")
print(f"Interactions train : {len(train):,}")
print(f"Interactions test : {len(test):,}")


# --------------------------------------------------
# CHARGEMENT DES MODELES
# --------------------------------------------------

print("\nChargement des trois modèles...")

with open(CONTENT_MODEL_PATH, "rb") as f:
    content_model = joblib.load(f)

with open(SVD_MODEL_PATH, "rb") as f:
    svd_model = joblib.load(f)

with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_model = joblib.load(f)

print("Les trois modèles sont chargés avec succès.")

print(f"Recettes Content-Based : {len(content_model.recipe_to_index):,}")
print(f"Recettes SVD : {len(svd_model.item_to_index):,}")


# --------------------------------------------------
# PREPARATION DES UTILISATEURS
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

eligible_users = []

for user_id, user_data in train_positive.groupby("AuthorId"):

    known_recipes = user_data["RecipeId"].unique()

    # Le split conserve exactement 5 recettes positives connues
    if len(known_recipes) == 5:

        # L'utilisateur doit aussi avoir des recettes positives masquées
        hidden_recipes = test_positive[
            test_positive["AuthorId"] == user_id
        ]

        if not hidden_recipes.empty:
            eligible_users.append(user_id)


random.shuffle(eligible_users)

selected_users = eligible_users[:N_USERS]

if not selected_users:
    raise ValueError("Aucun utilisateur éligible pour le test.")

print(f"\nNombre d'utilisateurs sélectionnés : {len(selected_users)}")

if len(selected_users) < N_USERS:
    print("Attention : moins de 500 utilisateurs éligibles disponibles.")


# --------------------------------------------------
# CATALOGUE COMMUN
# --------------------------------------------------

# Le SVD recommande uniquement des recettes présentes dans son train.
# On utilise ce catalogue comme référence commune.
common_catalog = set(svd_model.item_to_index.keys())

print(f"Catalogue commun de référence : {len(common_catalog):,} recettes")


# --------------------------------------------------
# FONCTION D'EVALUATION
# --------------------------------------------------

def evaluate_metrics(recommended_ids, relevant_ids):

    recommended_ids = list(dict.fromkeys(
        str(recipe_id) for recipe_id in recommended_ids
    ))[:TOP_K]

    relevant_ids = set(str(recipe_id) for recipe_id in relevant_ids)

    relevance = [
        1 if recipe_id in relevant_ids else 0
        for recipe_id in recommended_ids
    ]

    hits = sum(relevance)

    # Precision@K
    precision = hits / TOP_K

    # Recall@K
    recall = (
        hits / len(relevant_ids)
        if relevant_ids
        else 0
    )

    # NDCG@K
    dcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(relevance, start=1)
    )

    ideal_relevance = [1] * min(len(relevant_ids), TOP_K)

    idcg = sum(
        rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_relevance, start=1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0

    # HitRate@K
    hit_rate = 1 if hits > 0 else 0

    return hits, precision, recall, ndcg, hit_rate, len(recommended_ids)


# --------------------------------------------------
# EVALUATION DES TROIS MODELES
# --------------------------------------------------

results = []

models = [
    ("Content-Based", content_model),
    ("SVD", svd_model),
    ("Hybrid", hybrid_model)
]

for model_name, model in models:

    print("\n" + "=" * 55)
    print(f"EVALUATION : {model_name}")
    print("=" * 55)

    for test_number, user_id in enumerate(selected_users, start=1):

        # Les 5 mêmes recettes connues pour les trois modèles
        user_train = train_positive[
            train_positive["AuthorId"] == user_id
        ].sort_values("RecipeId")

        liked_ids = user_train["RecipeId"].tolist()
        liked_ratings = user_train["Rating"].tolist()

        # Recettes positives masquées
        user_test = test_positive[
            test_positive["AuthorId"] == user_id
        ]

        relevant_ids = set(
            user_test["RecipeId"].astype(str)
        )

        # Évaluation uniquement sur les recettes du catalogue commun
        relevant_ids = relevant_ids.intersection(common_catalog)

        if not relevant_ids:
            continue

        try:

            # ----------------------------------------
            # CONTENT-BASED
            # ----------------------------------------

            if model_name == "Content-Based":

                # Demander un pool plus grand pour filtrer
                # les recettes absentes du catalogue commun.
                recommendations = model.recommend(
                    liked_recipe_ids=liked_ids,
                    top_n=500
                )

                if recommendations.empty:
                    recommended_ids = []
                else:
                    recommended_ids = [
                        recipe_id
                        for recipe_id in recommendations["RecipeId"].astype(str)
                        if recipe_id in common_catalog
                    ][:TOP_K]

            # ----------------------------------------
            # SVD
            # ----------------------------------------

            elif model_name == "SVD":

                recommendations = model.recommend(
                    liked_recipe_ids=liked_ids,
                    liked_ratings=liked_ratings,
                    top_n=TOP_K
                )

                recommended_ids = (
                    recommendations["RecipeId"].astype(str).tolist()
                    if not recommendations.empty
                    else []
                )

            # ----------------------------------------
            # HYBRID
            # ----------------------------------------

            elif model_name == "Hybrid":

                recommendations = model.recommend(
                    liked_recipe_ids=liked_ids,
                    liked_ratings=liked_ratings,
                    top_n=TOP_K
                )

                recommended_ids = (
                    recommendations["RecipeId"].astype(str).tolist()
                    if not recommendations.empty
                    else []
                )

            # ----------------------------------------
            # CALCUL DES METRIQUES
            # ----------------------------------------

            (
                hits,
                precision,
                recall,
                ndcg,
                hit_rate,
                returned
            ) = evaluate_metrics(recommended_ids, relevant_ids)

            results.append({
                "Model": model_name,
                "UserId": user_id,
                "KnownLikes": len(liked_ids),
                "HiddenRecipes": len(relevant_ids),
                "ReturnedRecipes": returned,
                "Hits": hits,
                "Precision@10": precision,
                "Recall@10": recall,
                "NDCG@10": ndcg,
                "HitRate@10": hit_rate
            })

        except Exception as error:
            print(
                f"\nErreur {model_name} - utilisateur {user_id} : {error}"
            )

        if test_number % 25 == 0 or test_number == len(selected_users):
            print(
                f"{test_number}/{len(selected_users)} utilisateurs traités"
            )


# --------------------------------------------------
# RESULTATS
# --------------------------------------------------

if not results:
    raise ValueError("Aucun résultat d'évaluation obtenu.")

results_df = pd.DataFrame(results)

summary = (
    results_df.groupby("Model")[
        [
            "Precision@10",
            "Recall@10",
            "NDCG@10",
            "HitRate@10"
        ]
    ]
    .mean()
)

print("\n" + "=" * 55)
print("RESULTATS DE LA COMPARAISON")
print("=" * 55)

print(f"\nNombre d'utilisateurs sélectionnés : {len(selected_users)}")

print("\nRésultats moyens par modèle :")
print(summary.round(4).to_string())


# --------------------------------------------------
# SAUVEGARDE
# --------------------------------------------------

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nRésultats détaillés sauvegardés : {RESULTS_PATH}")

# Résumé séparé
summary_path = ROOT_DIR / "tests" / "recommendation_models_summary.csv"

summary.to_csv(summary_path)

print(f"Résumé sauvegardé : {summary_path}")

print("\nÉvaluation terminée.")
