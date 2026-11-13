import sys
import pickle
import random
import numpy as np
import pandas as pd

from pathlib import Path

# Ajouter la racine du projet au chemin Python
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Import nécessaire pour utiliser le modèle existant
from src.recommendation.preference_clustering import recommend_for_user


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = ROOT_DIR / "models" / "preference_model.pkl"
RESULTS_PATH = ROOT_DIR / "tests" / "preference_clustering_test_results.csv"

N_USERS = 500
TOP_K = 10
RANDOM_SEED = 42
TEST_RATIO = 0.30


# --------------------------------------------------
# CHARGEMENT DU MODELE
# --------------------------------------------------

print("Chargement du modèle Preference Clustering...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

interactions = model["interactions"].tocsr()
recipe_ids = model["recipe_ids"]

print("\nModèle chargé avec succès.")
print(f"Nombre d'utilisateurs : {interactions.shape[0]:,}")
print(f"Nombre de recettes : {interactions.shape[1]:,}")
print(f"Nombre d'interactions : {interactions.nnz:,}")
print(f"Nombre de clusters : {model['config']['n_clusters']}")


# --------------------------------------------------
# PREPARATION DES UTILISATEURS
# --------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

eligible_users = []

for user_idx in range(interactions.shape[0]):

    user_recipes = interactions.getrow(user_idx).indices

    # Au moins deux recettes pour séparer train et test
    if len(user_recipes) >= 2:
        eligible_users.append(user_idx)

random.shuffle(eligible_users)

selected_users = eligible_users[:N_USERS]

if not selected_users:
    raise ValueError("Aucun utilisateur éligible pour le test.")

print(f"\nNombre d'utilisateurs sélectionnés : {len(selected_users)}")

if len(selected_users) < N_USERS:
    print("Attention : moins de 500 utilisateurs éligibles disponibles.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

results = []

precisions = []
recalls = []
ndcgs = []
hit_rates = []

print("\nDébut de l'évaluation...\n")

for test_number, user_idx in enumerate(selected_users, start=1):

    user_recipes = interactions.getrow(user_idx).indices.copy()

    random.shuffle(user_recipes)

    # Séparation des interactions connues et masquées
    n_test = max(1, int(len(user_recipes) * TEST_RATIO))

    test_indices = user_recipes[:n_test]
    liked_indices = user_recipes[n_test:]

    if len(liked_indices) == 0:
        continue

    # Recettes masquées considérées comme pertinentes
    relevant_ids = {
        int(recipe_ids[idx])
        for idx in test_indices
    }

    # Exclure les recettes déjà vues et les likes connus
    seen_indices = set(liked_indices.tolist())

    # Génération des recommandations avec le modèle existant
    recommended_indices, cluster_id = recommend_for_user(
        liked_indices=liked_indices.tolist(),
        disliked_indices=[],
        model_data=model,
        top_n=TOP_K,
        seen_indices=seen_indices
    )

    recommended_ids = [
        int(recipe_ids[idx])
        for idx in recommended_indices
    ]

    # Pertinence des recommandations
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

    precisions.append(precision)
    recalls.append(recall)
    ndcgs.append(ndcg)
    hit_rates.append(hit_rate)

    results.append({
        "UserIndex": user_idx,
        "KnownLikes": len(liked_indices),
        "HiddenRecipes": len(test_indices),
        "ReturnedRecipes": len(recommended_ids),
        "Hits": hits,
        "ClusterID": cluster_id,
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg,
        "HitRate@10": hit_rate
    })

    if test_number % 25 == 0 or test_number == len(selected_users):
        print(
            f"{test_number}/{len(selected_users)} utilisateurs traités"
        )


# --------------------------------------------------
# RESULTATS
# --------------------------------------------------

if not results:
    raise ValueError("Aucun résultat d'évaluation obtenu.")

summary = {
    "NumberOfUsers": len(results),
    "Precision@10": np.mean(precisions),
    "Recall@10": np.mean(recalls),
    "NDCG@10": np.mean(ndcgs),
    "HitRate@10": np.mean(hit_rates)
}

print("\n" + "=" * 50)
print("RESULTATS DE L'EVALUATION")
print("=" * 50)

for metric, value in summary.items():

    if isinstance(value, (float, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")


# --------------------------------------------------
# SAUVEGARDE
# --------------------------------------------------

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

pd.DataFrame(results).to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nRésultats détaillés sauvegardés : {RESULTS_PATH}")
