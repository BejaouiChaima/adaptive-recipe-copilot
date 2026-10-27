
# src/recommendation/content_based_svd.py

import ast
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedSVDRecommender:
    def __init__(
        self,
        recipes_path="data/raw/recipes.csv",
        reviews_path="data/raw/reviews.csv",
        max_recipes=60000,
        max_users=2500,
        n_components=50,
        content_weight=0.5,
        random_state=42,
    ):
        self.recipes_path = recipes_path
        self.reviews_path = reviews_path
        self.max_recipes = max_recipes
        self.max_users = max_users
        self.n_components = n_components
        self.content_weight = content_weight
        self.random_state = random_state

    @staticmethod
    def _parse_text(value):
        if value is None or pd.isna(value):
            return ""

        if isinstance(value, (list, tuple, set)):
            return " ".join(map(str, value))

        value = str(value)

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return " ".join(map(str, parsed))
        except (ValueError, SyntaxError):
            pass

        return value

    @staticmethod
    def _rank_scores(scores):
        """Convertit les scores en scores de classement comparables."""
        scores = np.nan_to_num(
            np.asarray(scores, dtype=np.float64),
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )

        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int32)
        ranks[order] = np.arange(len(scores))

        if len(scores) <= 1:
            return np.ones(len(scores), dtype=np.float32)

        return (1.0 - ranks / (len(scores) - 1)).astype(np.float32)

    def load(self):
        print("Chargement des recettes...")
        recipes = pd.read_csv(self.recipes_path, low_memory=False)

        print("Chargement des avis...")
        reviews = pd.read_csv(self.reviews_path, low_memory=False)

        required_recipes = {"RecipeId", "Name"}
        required_reviews = {"RecipeId", "AuthorId", "Rating"}

        if not required_recipes.issubset(recipes.columns):
            raise ValueError(
                f"Colonnes manquantes dans recipes.csv : "
                f"{required_recipes - set(recipes.columns)}"
            )

        if not required_reviews.issubset(reviews.columns):
            raise ValueError(
                f"Colonnes manquantes dans reviews.csv : "
                f"{required_reviews - set(reviews.columns)}"
            )

        # Identifiants homogènes pour éviter les problèmes int/string.
        recipes["RecipeId"] = recipes["RecipeId"].astype(str)
        reviews["RecipeId"] = reviews["RecipeId"].astype(str)
        reviews["AuthorId"] = reviews["AuthorId"].astype(str)
        reviews["Rating"] = pd.to_numeric(
            reviews["Rating"], errors="coerce"
        )

        reviews = reviews.dropna(
            subset=["RecipeId", "AuthorId", "Rating"]
        )
        reviews = reviews[reviews["Rating"].between(1, 5)]

        # Agréger les avis répétés d'un même utilisateur pour une recette.
        reviews = (
            reviews.groupby(["AuthorId", "RecipeId"], as_index=False)
            ["Rating"].mean()
        )

        # Limitation des recettes : les plus commentées.
        popular_recipe_ids = (
            reviews["RecipeId"]
            .value_counts()
            .head(self.max_recipes)
            .index
        )

        recipes = recipes[
            recipes["RecipeId"].isin(popular_recipe_ids)
        ].drop_duplicates("RecipeId").copy()

        reviews = reviews[
            reviews["RecipeId"].isin(recipes["RecipeId"])
        ].copy()

        # Limitation des utilisateurs les plus actifs.
        active_user_ids = (
            reviews["AuthorId"]
            .value_counts()
            .head(self.max_users)
            .index
        )

        reviews = reviews[
            reviews["AuthorId"].isin(active_user_ids)
        ].copy()

        # Champs textuels disponibles dans le fichier.
        text_columns = [
            column for column in [
                "Name",
                "RecipeCategory",
                "Keywords",
                "Description",
                "RecipeIngredientParts",
            ]
            if column in recipes.columns
        ]

        for column in text_columns:
            recipes[column] = recipes[column].apply(self._parse_text)

        recipes["recipe_text"] = (
            recipes[text_columns]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.strip()
        )

        recipes = recipes.reset_index(drop=True)

        self.recipes = recipes
        self.reviews = reviews
        self.recipe_ids = recipes["RecipeId"].tolist()
        self.recipe_to_idx = {
            recipe_id: idx
            for idx, recipe_id in enumerate(self.recipe_ids)
        }

        # Construire TF-IDF.
        print("Construction des vecteurs TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=80000,
            ngram_range=(1, 2),
            min_df=2,
            dtype=np.float32,
        )

        self.recipe_vectors = self.vectorizer.fit_transform(
            self.recipes["recipe_text"]
        )

        print(f"Recettes : {len(self.recipes):,}")
        print(f"Utilisateurs : {self.reviews['AuthorId'].nunique():,}")
        print(f"Interactions : {len(self.reviews):,}")
        print("Chargement terminé.")

        return self

    def _fit_svd(self, training_reviews):
        """
        SVD sur les notes centrées par utilisateur.
        Les utilisateurs de test ne doivent pas apparaître ici.
        """
        training_reviews = training_reviews.copy()

        if training_reviews.empty:
            raise ValueError("Le jeu d'entraînement est vide.")

        user_means = training_reviews.groupby("AuthorId")[
            "Rating"
        ].transform("mean")

        # Les notes représentent l'écart à la moyenne de l'utilisateur.
        training_reviews["CenteredRating"] = (
            training_reviews["Rating"] - user_means
        )

        user_ids = training_reviews["AuthorId"].unique().tolist()
        user_to_idx = {
            user_id: idx for idx, user_id in enumerate(user_ids)
        }

        rows = training_reviews["AuthorId"].map(
            user_to_idx
        ).to_numpy()

        cols = training_reviews["RecipeId"].map(
            self.recipe_to_idx
        ).to_numpy()

        values = training_reviews["CenteredRating"].to_numpy(
            dtype=np.float32
        )

        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(user_ids), len(self.recipe_ids)),
            dtype=np.float32,
        )

        # Limiter le nombre de composantes aux dimensions disponibles.
        n_components = min(
            self.n_components,
            matrix.shape[0] - 1,
            matrix.shape[1] - 1,
        )

        if n_components < 1:
            raise ValueError("Dimensions insuffisantes pour SVD.")

        print(
            f"Entraînement SVD : {matrix.shape[0]} utilisateurs, "
            f"{matrix.shape[1]} recettes, "
            f"{n_components} composantes..."
        )

        self.svd = TruncatedSVD(
            n_components=n_components,
            n_iter=7,
            random_state=self.random_state,
        )

        self.svd.fit(matrix)

        # Shape : (composantes latentes, recettes).
        self.item_factors = self.svd.components_.astype(np.float32)

    def _content_scores(self, liked_indices):
        # Moyenne des vecteurs TF-IDF des recettes aimées.
        profile = self.recipe_vectors[liked_indices].mean(axis=0)

        return cosine_similarity(
            csr_matrix(profile),
            self.recipe_vectors,
        ).ravel()

    def _svd_scores(self, liked_indices):
        """
        Estime un profil latent pour le nouvel utilisateur à partir
        des seules recettes qu'il aime, avec régularisation ridge.
        """
        item_factors = self.item_factors

        # Une ligne par recette aimée, une colonne par facteur latent.
        X = item_factors[:, liked_indices].T.astype(np.float64)

        # Le profil indique que les recettes aimées sont positives.
        y = np.ones(len(liked_indices), dtype=np.float64)

        n_factors = X.shape[1]
        alpha = 1.0

        # Résolution régularisée du profil latent :
        # (X'X + alpha I)u = X'y
        A = X.T @ X + alpha * np.eye(n_factors)
        b = X.T @ y

        try:
            user_factors = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            user_factors = np.linalg.lstsq(A, b, rcond=None)[0]

        return (user_factors @ item_factors).ravel()

    def recommend_from_likes(
        self,
        liked_recipe_ids,
        top_n=10,
        method="hybrid",
    ):
        liked_ids = list(dict.fromkeys(
            str(recipe_id) for recipe_id in liked_recipe_ids
        ))

        liked_indices = [
            self.recipe_to_idx[recipe_id]
            for recipe_id in liked_ids
            if recipe_id in self.recipe_to_idx
        ]

        if not liked_indices:
            return []

        content_scores = self._content_scores(liked_indices)

        if method == "content":
            final_scores = self._rank_scores(content_scores)

        elif method == "hybrid":
            svd_scores = self._svd_scores(liked_indices)

            content_rank = self._rank_scores(content_scores)
            svd_rank = self._rank_scores(svd_scores)

            weight = self.content_weight
            final_scores = (
                weight * content_rank
                + (1.0 - weight) * svd_rank
            )

        else:
            raise ValueError(
                "method doit être 'content' ou 'hybrid'."
            )

        # Exclure les recettes déjà aimées.
        final_scores[liked_indices] = -np.inf

        top_indices = np.argsort(-final_scores)[:top_n]

        return [self.recipe_ids[idx] for idx in top_indices]

    @staticmethod
    def _metrics(recommended, relevant, top_k):
        recommended = recommended[:top_k]
        relevant = set(relevant)
        hits = set(recommended) & relevant

        precision = len(hits) / top_k if top_k else 0.0
        recall = len(hits) / len(relevant) if relevant else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        # NDCG@K : récompense davantage les résultats pertinents en haut.
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, recipe_id in enumerate(recommended)
            if recipe_id in relevant
        )

        ideal_hits = min(len(relevant), top_k)
        idcg = sum(
            1.0 / np.log2(rank + 2)
            for rank in range(ideal_hits)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "Hits": len(hits),
            "Precision@K": precision,
            "Recall@K": recall,
            "F1@K": f1,
            "NDCG@K": ndcg,
        }

    def evaluate_cold_start(
        self,
        n_users=50,
        n_likes=3,
        n_test=3,
        top_k=10,
        random_state=42,
    ):
        rng = np.random.default_rng(random_state)

        # Les notes >= 4 sont considérées comme positives.
        positive = self.reviews[
            self.reviews["Rating"] >= 4
        ].copy()

        positive = positive[
            positive["RecipeId"].isin(self.recipe_to_idx)
        ]

        positives_by_user = (
            positive.groupby("AuthorId")["RecipeId"]
            .apply(lambda s: s.drop_duplicates().tolist())
        )

        eligible_users = positives_by_user[
            positives_by_user.apply(
                lambda ids: len(ids) >= n_likes + n_test
            )
        ].index.to_numpy()

        if len(eligible_users) == 0:
            raise ValueError(
                "Aucun utilisateur ne possède assez de recettes "
                "positives pour ce test."
            )

        selected_users = rng.choice(
            eligible_users,
            size=min(n_users, len(eligible_users)),
            replace=False,
        ).tolist()

        # Point crucial : toutes les interactions de ces utilisateurs
        # sont exclues du SVD d'entraînement.
        training_reviews = self.reviews[
            ~self.reviews["AuthorId"].isin(selected_users)
        ].copy()

        self._fit_svd(training_reviews)

        rows = []

        for user_id in selected_users:
            user_positives = np.asarray(
                positives_by_user.loc[user_id],
                dtype=object,
            )

            shuffled = rng.permutation(user_positives)

            liked_ids = shuffled[:n_likes].tolist()
            relevant_ids = shuffled[
                n_likes:n_likes + n_test
            ].tolist()

            content_recs = self.recommend_from_likes(
                liked_ids,
                top_n=top_k,
                method="content",
            )

            hybrid_recs = self.recommend_from_likes(
                liked_ids,
                top_n=top_k,
                method="hybrid",
            )

            content_metrics = self._metrics(
                content_recs, relevant_ids, top_k
            )
            hybrid_metrics = self._metrics(
                hybrid_recs, relevant_ids, top_k
            )

            rows.append({
                "AuthorId": user_id,
                "N_Likes": len(liked_ids),
                "N_Hidden": len(relevant_ids),
                "LikedRecipeIds": liked_ids,
                "HiddenRecipeIds": relevant_ids,
                "ContentRecommendations": content_recs,
                "HybridRecommendations": hybrid_recs,
                **{
                    f"Content_{key}": value
                    for key, value in content_metrics.items()
                },
                **{
                    f"Hybrid_{key}": value
                    for key, value in hybrid_metrics.items()
                },
            })

        details = pd.DataFrame(rows)

        metric_columns = [
            "Precision@K", "Recall@K", "F1@K", "NDCG@K"
        ]

        summary_rows = []

        for method in ["Content", "Hybrid"]:
            prefix = f"{method}_"
            summary = {
                "Model": (
                    "Content-Based TF-IDF"
                    if method == "Content"
                    else "Content-Based + SVD"
                ),
                "UsersEvaluated": len(details),
                "UsersWithHit": int(
                    (details[f"{prefix}Hits"] > 0).sum()
                ),
                "TotalHits": int(details[f"{prefix}Hits"].sum()),
            }

            for metric in metric_columns:
                summary[metric] = details[
                    f"{prefix}{metric}"
                ].mean()

            summary_rows.append(summary)

        return pd.DataFrame(summary_rows), details
