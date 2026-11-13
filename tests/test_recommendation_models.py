
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm


# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_SOURCE_DIR = PROJECT_ROOT / "src" / "models"
sys.path.insert(0, str(MODELS_SOURCE_DIR))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models" / "recommendation"
DATA_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

TRAIN_PATH = DATA_DIR / "train_reviews.csv"
TEST_PATH = DATA_DIR / "test_reviews.csv"

CONTENT_MODEL_PATH = MODELS_DIR / "content_model.pkl"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.pkl"

TOP_K = 10
MAX_USERS = 500
RANDOM_STATE = 42

# ============================================================
# CHARGEMENT DES DONNÉES ET MODÈLES
# ============================================================

def load_data():
    print("\nChargement des données...")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for df in (train, test):
        df["AuthorId"] = df["AuthorId"].astype(str)
        df["RecipeId"] = df["RecipeId"].astype(str)
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    train = train.dropna(subset=["AuthorId", "RecipeId", "Rating"])
    test = test.dropna(subset=["AuthorId", "RecipeId", "Rating"])

    return train, test


def load_models():
    print("\nChargement des trois modèles...")

    content_model = joblib.load(CONTENT_MODEL_PATH)
    svd_model = joblib.load(SVD_MODEL_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)

    print("Les trois modèles ont été chargés.")

    return content_model, svd_model, hybrid_model


# ============================================================
# MÉTRIQUES
# ============================================================

def calculate_metrics(recommended_ids, relevant_ids):
    recommended_ids = list(dict.fromkeys(map(str, recommended_ids)))[:TOP_K]
    relevant_ids = set(map(str, relevant_ids))

    hits = len(set(recommended_ids) & relevant_ids)

    precision = hits / TOP_K if TOP_K else 0
    recall = hits / len(relevant_ids) if relevant_ids else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    dcg = sum(
        1 / np.log2(rank + 2)
        for rank, recipe_id in enumerate(recommended_ids)
        if recipe_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), TOP_K)
    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg > 0 else 0
    hit_rate = 1 if hits > 0 else 0

    return {
        "Precision@10": precision,
        "Recall@10": recall,
        "F1@10": f1,
        "NDCG@10": ndcg,
        "HitRate@10": hit_rate,
    }


# ============================================================
# PRÉPARATION DES UTILISATEURS
# ============================================================

def prepare_users(train, test, svd_model):

    train_positive = train[train["Rating"] >= 4].copy()
    test_positive = test[test["Rating"] >= 4].copy()

    # Utilisateurs ayant exactement 5 recettes positives connues
    known_counts = train_positive.groupby("AuthorId")["RecipeId"].nunique()
    eligible_users = known_counts[known_counts == 5].index

    # Catalogue commun disponible dans le modèle SVD
    common_catalog = set(map(str, svd_model.item_to_index.keys()))

    # Recettes pertinentes présentes dans le catalogue commun
    test_positive = test_positive[
        test_positive["RecipeId"].isin(common_catalog)
    ]

    users_with_test = set(test_positive["AuthorId"].unique())
    eligible_users = [
        user for user in eligible_users
        if user in users_with_test
    ]

    rng = np.random.RandomState(RANDOM_STATE)

    if len(eligible_users) > MAX_USERS:
        eligible_users = list(
            rng.choice(eligible_users, MAX_USERS, replace=False)
        )

    print(f"Nombre d'utilisateurs évalués : {len(eligible_users)}")

    return train_positive, test_positive, eligible_users, common_catalog


# ============================================================
# ÉVALUATION D'UN MODÈLE
# ============================================================

def evaluate_model(model_name, model, train_positive,
                   test_positive, users, common_catalog):

    print(f"\nÉvaluation du modèle : {model_name}")

    results = []

    for user_id in tqdm(users):

        user_train = train_positive[
            train_positive["AuthorId"] == user_id
        ].copy()

        # Les 5 mêmes recettes connues pour chaque modèle
        user_train = user_train.sort_values("RecipeId")

        liked_ids = user_train["RecipeId"].tolist()
        liked_ratings = user_train["Rating"].tolist()

        relevant_ids = test_positive[
            test_positive["AuthorId"] == user_id
        ]["RecipeId"].tolist()

        if not liked_ids or not relevant_ids:
            continue

        try:
            if model_name == "Content-Based":
                recommendations = model.recommend(
                    liked_ids,
                    top_n=TOP_K
                )

            elif model_name == "SVD":
                recommendations = model.recommend(
                    liked_ids,
                    liked_ratings,
                    top_n=TOP_K
                )

            elif model_name == "Hybrid":
                recommendations = model.recommend(
                    liked_ids,
                    liked_ratings,
                    top_n=TOP_K
                )

            else:
                raise ValueError(f"Modèle inconnu : {model_name}")

            if recommendations is None or recommendations.empty:
                recommended_ids = []
            else:
                recommended_ids = recommendations["RecipeId"].astype(str).tolist()

            metrics = calculate_metrics(recommended_ids, relevant_ids)

            results.append({
                "UserId": user_id,
                "Model": model_name,
                **metrics
            })

        except Exception as error:
            print(f"\nErreur pour {model_name}, utilisateur {user_id} : {error}")

    return pd.DataFrame(results)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("COMPARAISON DES TROIS MODÈLES DE RECOMMANDATION")
    print("=" * 60)

    # Vérification des fichiers
    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        CONTENT_MODEL_PATH,
        SVD_MODEL_PATH,
        HYBRID_MODEL_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}"
            )

    train, test = load_data()

    content_model, svd_model, hybrid_model = load_models()

    train_positive, test_positive, users, common_catalog = prepare_users(
        train, test, svd_model
    )

    if not users:
        raise ValueError(
            "Aucun utilisateur éligible trouvé. Vérifie les fichiers train/test."
        )

    # Évaluation des trois modèles
    content_results = evaluate_model(
        "Content-Based", content_model,
        train_positive, test_positive, users, common_catalog
    )

    svd_results = evaluate_model(
        "SVD", svd_model,
        train_positive, test_positive, users, common_catalog
    )

    hybrid_results = evaluate_model(
        "Hybrid", hybrid_model,
        train_positive, test_positive, users, common_catalog
    )

    all_results = pd.concat(
        [content_results, svd_results, hybrid_results],
        ignore_index=True
    )

    if all_results.empty:
        raise ValueError("Aucun résultat d'évaluation n'a été généré.")

    # Résultats détaillés
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = RESULTS_DIR / "three_models_detailed_results.csv"
    all_results.to_csv(detail_path, index=False)

    # Moyenne des métriques par modèle
    comparison = (
        all_results.groupby("Model")[
            ["Precision@10", "Recall@10", "F1@10",
             "NDCG@10", "HitRate@10"]
        ]
        .mean()
        .reset_index()
    )

    comparison_path = RESULTS_DIR / "three_models_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 60)
    print("RÉSULTATS DE LA COMPARAISON")
    print("=" * 60)

    print(comparison.to_string(index=False))

    print("\nFichiers générés :")
    print(detail_path)
    print(comparison_path)

    print("\nÉvaluation terminée.")


if __name__ == "__main__":
    main()
