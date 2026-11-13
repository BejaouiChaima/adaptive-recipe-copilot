
from pathlib import Path
import gc
import pickle

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MAX_USERS = None
MAX_RECIPES = None

N_CLUSTERS = 30
MAX_SIMILAR_RECIPES = 50

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "preference_model.pkl"


# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

def load_data():

    print("Chargement des reviews...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    reviews = pd.read_csv(
        DATA_PATH,
        usecols=["AuthorId", "RecipeId", "Rating"]
    )

    reviews = reviews.dropna(
        subset=["AuthorId", "RecipeId", "Rating"]
    )

    # On conserve les interactions positives.
    reviews = reviews[reviews["Rating"] >= 4].copy()

    reviews = reviews.drop_duplicates(
        subset=["AuthorId", "RecipeId"]
    )

    if reviews.empty:
        raise ValueError("Aucune interaction positive trouvée.")

    print(f"Interactions positives : {len(reviews):,}")

    # Utilisateurs ayant au moins deux interactions positives.
    user_counts = (
        reviews.groupby("AuthorId")["RecipeId"]
        .nunique()
    )

    eligible_users = user_counts[
        user_counts >= 2
    ].index

    reviews = reviews[
        reviews["AuthorId"].isin(eligible_users)
    ].copy()

    del user_counts, eligible_users
    gc.collect()

    # Limitation facultative du nombre d'utilisateurs.
    if MAX_USERS is not None:
        selected_users = (
            reviews["AuthorId"]
            .drop_duplicates()
            .head(MAX_USERS)
        )

        reviews = reviews[
            reviews["AuthorId"].isin(selected_users)
        ].copy()

    # Catalogue de recettes.
    recipe_counts = reviews["RecipeId"].value_counts()

    if MAX_RECIPES is not None:
        selected_recipes = recipe_counts.head(MAX_RECIPES).index
        reviews = reviews[
            reviews["RecipeId"].isin(selected_recipes)
        ].copy()

    recipe_ids = (
        reviews["RecipeId"]
        .drop_duplicates()
        .tolist()
    )

    user_ids = (
        reviews["AuthorId"]
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

    rows = reviews["AuthorId"].map(user_to_idx).to_numpy(
        dtype=np.int32
    )

    cols = reviews["RecipeId"].map(recipe_to_idx).to_numpy(
        dtype=np.int32
    )

    interactions = csr_matrix(
        (
            np.ones(len(reviews), dtype=np.float32),
            (rows, cols)
        ),
        shape=(len(user_ids), len(recipe_ids)),
        dtype=np.float32
    )

    print("\nDonnées chargées :")
    print(f"Utilisateurs : {interactions.shape[0]:,}")
    print(f"Recettes : {interactions.shape[1]:,}")
    print(f"Interactions : {interactions.nnz:,}")

    del reviews, rows, cols
    gc.collect()

    return interactions, recipe_ids


# ============================================================
# 2. SIMILARITES ENTRE RECETTES
# ============================================================

def build_item_similarity(normalized_items):

    n_recipes = normalized_items.shape[0]

    print("\nCalcul des similarités entre recettes...")
    print(f"Nombre de recettes : {n_recipes:,}")

    rows = []
    cols = []
    values = []

    normalized_items_T = normalized_items.T.tocsr()

    for recipe_idx in range(n_recipes):

        similarity_row = (
            normalized_items.getrow(recipe_idx)
            @ normalized_items_T
        ).tocsr()

        indices = similarity_row.indices
        scores = similarity_row.data

        # Exclure la recette elle-même et les scores nuls.
        mask = (
            (indices != recipe_idx)
            & (scores > 0)
        )

        indices = indices[mask]
        scores = scores[mask]

        if len(scores) > MAX_SIMILAR_RECIPES:

            selected = np.argpartition(
                scores,
                -MAX_SIMILAR_RECIPES
            )[-MAX_SIMILAR_RECIPES:]

            indices = indices[selected]
            scores = scores[selected]

        rows.extend([recipe_idx] * len(indices))
        cols.extend(indices.tolist())
        values.extend(scores.tolist())

        if (recipe_idx + 1) % 1000 == 0:
            print(
                f"  {recipe_idx + 1:,}/{n_recipes:,} recettes"
            )

    item_similarity = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(rows, dtype=np.int32),
                np.asarray(cols, dtype=np.int32)
            )
        ),
        shape=(n_recipes, n_recipes),
        dtype=np.float32
    )

    item_similarity.eliminate_zeros()

    del normalized_items_T, rows, cols, values
    gc.collect()

    print(
        f"Similarités conservées : {item_similarity.nnz:,}"
    )

    return item_similarity


# ============================================================
# 3. ENTRAINEMENT
# ============================================================

def train_model(interactions):

    print("\nNormalisation des profils de recettes...")

    normalized_items = normalize(
        interactions.T,
        norm="l2",
        axis=1
    ).tocsr()

    item_similarity = build_item_similarity(
        normalized_items
    )

    del normalized_items
    gc.collect()

    print("\nConstruction des profils utilisateurs...")

    user_preference_matrix = (
        interactions @ item_similarity
    ).tocsr()

    user_preference_matrix = normalize(
        user_preference_matrix,
        norm="l2",
        axis=1
    ).tocsr()

    print(
        "Dimensions des profils :",
        user_preference_matrix.shape
    )

    # --------------------------------------------------------
    # CLUSTERING DES UTILISATEURS
    # --------------------------------------------------------

    n_clusters = min(
        N_CLUSTERS,
        interactions.shape[0]
    )

    print(f"\nEntraînement KMeans : {n_clusters} clusters")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=SEED,
        n_init=1,
        max_iter=100,
        algorithm="lloyd",
        verbose=1
    )

    labels = kmeans.fit_predict(user_preference_matrix)

    # --------------------------------------------------------
    # POPULARITE GLOBALE
    # --------------------------------------------------------

    global_popularity = np.asarray(
        interactions.sum(axis=0)
    ).ravel().astype(np.float32)

    if global_popularity.max() > 0:
        global_popularity /= global_popularity.max()

    # --------------------------------------------------------
    # POPULARITE PAR CLUSTER
    # --------------------------------------------------------

    cluster_popularity = {}

    print("\nCalcul de la popularité par cluster...")

    for cluster_id in range(n_clusters):

        members = np.flatnonzero(labels == cluster_id)

        if len(members) == 0:
            continue

        popularity = np.asarray(
            interactions[members].sum(axis=0)
        ).ravel().astype(np.float32)

        if popularity.max() > 0:
            popularity /= popularity.max()

        cluster_popularity[int(cluster_id)] = popularity

    print("\nEntraînement terminé.")

    return {
        "item_similarity": item_similarity,
        "kmeans": kmeans,
        "cluster_popularity": cluster_popularity,
        "global_popularity": global_popularity,
        "user_preference_matrix": user_preference_matrix,
        "interactions": interactions,
        "recipe_ids": None,
    }


# ============================================================
# 4. RECOMMANDATION PERSONNALISEE
# ============================================================

def recommend_for_user(
    liked_indices,
    disliked_indices,
    model_data,
    top_n=1,
    seen_indices=None
):

    interactions = model_data["interactions"]
    item_similarity = model_data["item_similarity"]
    kmeans = model_data["kmeans"]
    cluster_popularity = model_data["cluster_popularity"]
    global_popularity = model_data["global_popularity"]
    user_preference_matrix = model_data["user_preference_matrix"]

    n_recipes = interactions.shape[1]

    liked_indices = list(dict.fromkeys(
        int(i) for i in liked_indices
        if 0 <= int(i) < n_recipes
    ))

    disliked_indices = list(dict.fromkeys(
        int(i) for i in disliked_indices
        if 0 <= int(i) < n_recipes
    ))

    seen_indices = set(seen_indices or [])
    seen_indices.update(liked_indices)
    seen_indices.update(disliked_indices)

    # --------------------------------------------------------
    # COLD START : AUCUN LIKE
    # --------------------------------------------------------

    if not liked_indices:

        scores = global_popularity.copy()

        # Les dislikes réduisent le score des recettes similaires.
        if disliked_indices:

            disliked_profile = csr_matrix(
                (
                    np.ones(len(disliked_indices), dtype=np.float32),
                    (
                        np.zeros(len(disliked_indices), dtype=np.int32),
                        np.asarray(disliked_indices, dtype=np.int32)
                    )
                ),
                shape=(1, n_recipes)
            )

            negative_scores = (
                disliked_profile @ item_similarity
            ).toarray().ravel()

            if negative_scores.max() > 0:
                negative_scores /= negative_scores.max()

            scores -= 0.50 * negative_scores

        cluster_id = None

    else:

        # ----------------------------------------------------
        # PROFIL POSITIF DU NOUVEL UTILISATEUR
        # ----------------------------------------------------

        positive_profile = csr_matrix(
            (
                np.ones(len(liked_indices), dtype=np.float32),
                (
                    np.zeros(len(liked_indices), dtype=np.int32),
                    np.asarray(liked_indices, dtype=np.int32)
                )
            ),
            shape=(1, n_recipes)
        )

        positive_preferences = (
            positive_profile @ item_similarity
        ).toarray().ravel().astype(np.float32)

        # ----------------------------------------------------
        # PROFIL NEGATIF
        # ----------------------------------------------------

        negative_scores = np.zeros(
            n_recipes,
            dtype=np.float32
        )

        if disliked_indices:

            negative_profile = csr_matrix(
                (
                    np.ones(len(disliked_indices), dtype=np.float32),
                    (
                        np.zeros(len(disliked_indices), dtype=np.int32),
                        np.asarray(disliked_indices, dtype=np.int32)
                    )
                ),
                shape=(1, n_recipes)
            )

            negative_scores = (
                negative_profile @ item_similarity
            ).toarray().ravel().astype(np.float32)

        # ----------------------------------------------------
        # UTILISATEURS LES PLUS SIMILAIRES
        # ----------------------------------------------------

        normalized_profile = normalize(
            csr_matrix(positive_preferences.reshape(1, -1)),
            norm="l2"
        )

        similarities = (
            user_preference_matrix @ normalized_profile.T
        ).toarray().ravel()

        n_neighbors = min(50, len(similarities))

        neighbor_indices = np.argpartition(
            similarities,
            -n_neighbors
        )[-n_neighbors:]

        neighbor_weights = similarities[neighbor_indices]

        valid = neighbor_weights > 0

        neighbor_indices = neighbor_indices[valid]
        neighbor_weights = neighbor_weights[valid]

        if len(neighbor_indices) > 0:

            neighbor_interactions = interactions[
                neighbor_indices
            ]

            collaborative_scores = np.asarray(
                neighbor_interactions.T.dot(neighbor_weights)
            ).ravel()

            collaborative_scores /= (
                neighbor_weights.sum() + 1e-8
            )

        else:

            collaborative_scores = np.zeros(
                n_recipes,
                dtype=np.float32
            )

        if collaborative_scores.max() > 0:
            collaborative_scores /= collaborative_scores.max()

        # ----------------------------------------------------
        # CLUSTER LE PLUS PROCHE
        # ----------------------------------------------------

        normalized_centers = normalize(
            kmeans.cluster_centers_,
            norm="l2"
        )

        cluster_similarities = (
            normalized_profile @ normalized_centers.T
        ).ravel()

        cluster_id = int(np.argmax(cluster_similarities))

        cluster_scores = cluster_popularity[
            cluster_id
        ].copy()

        if cluster_scores.max() > 0:
            cluster_scores /= cluster_scores.max()

        # Normalisation du score personnel.
        if positive_preferences.max() > 0:
            positive_preferences /= positive_preferences.max()

        if negative_scores.max() > 0:
            negative_scores /= negative_scores.max()

        # ----------------------------------------------------
        # SCORE FINAL HYBRIDE
        # ----------------------------------------------------

        scores = (
            0.55 * collaborative_scores
            + 0.25 * positive_preferences
            + 0.20 * cluster_scores
            - 0.50 * negative_scores
        )

    # Ne jamais recommander une recette déjà vue ou évaluée.
    if seen_indices:
        scores[list(seen_indices)] = -np.inf

    available = np.flatnonzero(np.isfinite(scores))

    if len(available) == 0:
        return [], cluster_id

    count = min(top_n, len(available))

    selected = available[
        np.argpartition(scores[available], -count)[-count:]
    ]

    selected = selected[
        np.argsort(scores[selected])[::-1]
    ]

    return selected.tolist(), cluster_id


# ============================================================
# 5. SAUVEGARDE DU MODELE
# ============================================================

def main():

    np.random.seed(SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    interactions, recipe_ids = load_data()

    # On entraîne le modèle sur les interactions disponibles.
    model_data = train_model(interactions)

    # Ajouter les identifiants réels des recettes.
    model_data["recipe_ids"] = recipe_ids

    model_data["config"] = {
        "n_clusters": N_CLUSTERS,
        "max_similar_recipes": MAX_SIMILAR_RECIPES,
        "seed": SEED
    }

    print("\nSauvegarde du modèle...")

    temporary_path = MODEL_PATH.with_suffix(".tmp")

    with open(temporary_path, "wb") as file:
        pickle.dump(
            model_data,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    temporary_path.replace(MODEL_PATH)

    print("\nModèle sauvegardé avec succès.")
    print(f"Chemin : {MODEL_PATH.resolve()}")
    print(f"Taille : {MODEL_PATH.stat().st_size / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
