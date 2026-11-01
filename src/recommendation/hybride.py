from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = ROOT / "data" / "raw" / "reviews.csv"
RECIPES_PATH = ROOT / "data" / "raw" / "recipes.csv"


class ReviewTransferRecommender:
    """
    Recommandation par transfert de préférences :
    - crée un profil TF-IDF à partir des recettes aimées ;
    - recherche les profils historiques les plus similaires ;
    - recommande les recettes appréciées par ces utilisateurs.

    L'évaluation cold start simule un nouvel utilisateur :
    seules quelques recettes aimées lui sont montrées au modèle.
    D'autres recettes positives du même utilisateur sont gardées à part
    pour mesurer les résultats, et son profil historique est exclu des voisins.
    """

    def __init__(self, max_recipes=60000, max_users=2500):
        self.max_recipes = max_recipes
        self.max_users = max_users

    def load(self):
        print("Chargement des reviews...")

        reviews = pd.read_csv(
            REVIEWS_PATH,
            usecols=["AuthorId", "RecipeId", "Rating"],
        )

        reviews["Rating"] = pd.to_numeric(reviews["Rating"], errors="coerce")
        reviews = reviews.dropna(subset=["AuthorId", "RecipeId", "Rating"])
        reviews = reviews[reviews["Rating"].between(1, 5)].copy()
        reviews["AuthorId"] = reviews["AuthorId"].astype(str)
        reviews["RecipeId"] = reviews["RecipeId"].astype(str)
        reviews = reviews.drop_duplicates(["AuthorId", "RecipeId"], keep="last")

        # Limiter le corpus aux recettes les plus évaluées.
        popular_ids = reviews["RecipeId"].value_counts().head(
            self.max_recipes
        ).index
        reviews = reviews[reviews["RecipeId"].isin(popular_ids)]

        print("Chargement des informations des recettes...")
        recipes = pd.read_csv(
            RECIPES_PATH,
            usecols=[
                "RecipeId",
                "Name",
                "RecipeIngredientParts",
                "RecipeCategory",
                "Description",
            ],
            low_memory=False,
        )

        recipes["RecipeId"] = recipes["RecipeId"].astype(str)
        recipes = recipes[recipes["RecipeId"].isin(popular_ids)].copy()

        text_columns = [
            "Name",
            "RecipeIngredientParts",
            "RecipeCategory",
            "Description",
        ]
        for col in text_columns:
            recipes[col] = recipes[col].fillna("").astype(str)

        recipes = recipes.drop_duplicates("RecipeId").reset_index(drop=True)

        # Ne conserver que les reviews correspondant aux recettes disponibles.
        available_ids = set(recipes["RecipeId"])
        reviews = reviews[reviews["RecipeId"].isin(available_ids)].copy()

        # Garder les utilisateurs les plus actifs pour limiter le coût mémoire.
        user_counts = reviews["AuthorId"].value_counts()
        selected_users = user_counts.head(self.max_users).index
        reviews = reviews[reviews["AuthorId"].isin(selected_users)].copy()

        self.recipes = recipes
        self.recipe_names = dict(zip(recipes["RecipeId"], recipes["Name"]))
        self.recipe_to_idx = {
            rid: i for i, rid in enumerate(recipes["RecipeId"])
        }

        text = (
            recipes["Name"] + " "
            + recipes["RecipeIngredientParts"] + " "
            + recipes["RecipeCategory"] + " "
            + recipes["Description"]
        )

        print("Création des vecteurs des recettes...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            min_df=2,
            max_df=0.95,
            dtype=np.float32,
        )
        self.recipe_matrix = self.vectorizer.fit_transform(text)

        # Interactions conservées pour construire les profils et les recommandations.
        self.interactions = reviews[
            reviews["RecipeId"].isin(self.recipe_to_idx)
        ].copy()

        positive_reviews = self.interactions[
            self.interactions["Rating"] >= 4
        ]
        positive_counts = positive_reviews.groupby("AuthorId").size()
        eligible_users = positive_counts[positive_counts >= 3].index.tolist()

        print("Construction des profils des anciens utilisateurs...", flush=True)

        # Construction vectorisée des profils : on évite une boucle qui calcule
        # et stocke un vecteur dense séparé pour chaque utilisateur.
        self.profile_user_ids = [str(uid) for uid in eligible_users]
        user_to_idx = {
            uid: idx for idx, uid in enumerate(self.profile_user_ids)
        }

        positive_reviews = positive_reviews.copy()
        positive_reviews["UserIdx"] = positive_reviews["AuthorId"].map(user_to_idx)
        positive_reviews["RecipeIdx"] = positive_reviews["RecipeId"].map(
            self.recipe_to_idx
        )
        positive_reviews = positive_reviews.dropna(
            subset=["UserIdx", "RecipeIdx"]
        )
        positive_reviews["UserIdx"] = positive_reviews["UserIdx"].astype(np.int32)
        positive_reviews["RecipeIdx"] = positive_reviews["RecipeIdx"].astype(np.int32)
        weights = (
            positive_reviews["Rating"].to_numpy(dtype=np.float32) - 3.0
        )

        user_recipe_matrix = csr_matrix(
            (
                weights,
                (
                    positive_reviews["UserIdx"].to_numpy(),
                    positive_reviews["RecipeIdx"].to_numpy(),
                ),
            ),
            shape=(len(self.profile_user_ids), len(self.recipes)),
            dtype=np.float32,
        )

        # Chaque ligne est la somme pondérée des TF-IDF des recettes aimées.
        self.user_profiles = normalize(
            user_recipe_matrix @ self.recipe_matrix
        ).tocsr()

        self.interactions = self.interactions[
            self.interactions["AuthorId"].isin(self.profile_user_ids)
        ].copy()

        print(f"Recettes indexées : {len(self.recipes):,}")
        print(f"Profils historiques : {len(self.profile_user_ids):,}")
        print("Modèle prêt.")
        return self

    def popular_choices(self, limit=150):
        """Retourne des recettes populaires proposées au nouvel utilisateur."""
        ids = self.interactions["RecipeId"].value_counts().head(limit).index
        return self.recipes[
            self.recipes["RecipeId"].isin(ids)
        ][["RecipeId", "Name", "RecipeCategory"]].copy()

    def recommend_from_likes(
        self,
        liked_recipe_ids,
        top_n=10,
        exclude_author_id=None,
    ):
        """
        Recommande des recettes à partir des recettes aimées.

        exclude_author_id est utilisé pendant l'évaluation : l'utilisateur
        évalué ne doit pas faire partie des utilisateurs qui influencent
        ses propres recommandations.
        """
        liked_recipe_ids = list(dict.fromkeys(
            str(rid)
            for rid in liked_recipe_ids
            if str(rid) in self.recipe_to_idx
        ))

        if not liked_recipe_ids:
            return pd.DataFrame(), pd.DataFrame()

        indices = [self.recipe_to_idx[rid] for rid in liked_recipe_ids]

        # Profil provisoire du nouvel utilisateur.
        new_profile = csr_matrix(
            np.asarray(self.recipe_matrix[indices].mean(axis=0))
        )
        new_profile = normalize(new_profile)

        # Similarité avec les profils historiques.
        similarities = (
            self.user_profiles @ new_profile.T
        ).toarray().ravel()

        # Ne trier que les meilleurs candidats au lieu de trier tous les profils.
        if exclude_author_id is not None:
            excluded_idx = self.profile_user_ids.index(str(exclude_author_id)) \
                if str(exclude_author_id) in self.profile_user_ids else None
            if excluded_idx is not None:
                similarities[excluded_idx] = -1.0

        positive_positions = np.flatnonzero(similarities > 0)
        if len(positive_positions) == 0:
            return pd.DataFrame(), pd.DataFrame()

        candidate_positions = positive_positions
        if len(candidate_positions) > 20:
            local_top = np.argpartition(
                similarities[candidate_positions], -20
            )[-20:]
            candidate_positions = candidate_positions[local_top]
        best_positions = candidate_positions[
            np.argsort(similarities[candidate_positions])[::-1]
        ]

        neighbors = pd.DataFrame({
            "AuthorId": [self.profile_user_ids[i] for i in best_positions],
            "Similarity": [float(similarities[i]) for i in best_positions],
        })

        # Filtrer d'abord les interactions des seuls voisins : plus rapide que
        # de fusionner tout le DataFrame à chaque appel.
        similarity_by_user = dict(zip(neighbors["AuthorId"], neighbors["Similarity"]))
        evidence = self.interactions[
            self.interactions["AuthorId"].isin(similarity_by_user)
        ].copy()
        evidence["Similarity"] = evidence["AuthorId"].map(similarity_by_user)
        evidence = evidence.sort_values("Similarity", ascending=False)
        evidence = evidence[evidence["Rating"] >= 4]
        evidence = evidence[
            ~evidence["RecipeId"].isin(liked_recipe_ids)
        ].copy()

        if evidence.empty:
            return pd.DataFrame(), neighbors

        evidence["Contribution"] = (
            evidence["Similarity"] * (evidence["Rating"] - 3)
        )

        ranked = (
            evidence.groupby("RecipeId")
            .agg(
                Score=("Contribution", "sum"),
                SupportingReviews=("AuthorId", "nunique"),
            )
            .reset_index()
        )

        ranked = ranked.merge(
            self.recipes[
                [
                    "RecipeId",
                    "Name",
                    "RecipeCategory",
                    "RecipeIngredientParts",
                ]
            ],
            on="RecipeId",
            how="left",
        )

        ranked = ranked.sort_values(
            ["Score", "SupportingReviews"],
            ascending=False,
        ).head(top_n)

        # Conserver quelques avis comme explication de la recommandation.
        evidence_by_recipe = {}
        for rid, group in evidence.groupby("RecipeId"):
            top_evidence = group.sort_values(
                "Similarity", ascending=False
            ).head(3)
            evidence_by_recipe[rid] = "; ".join(
                f"User {row.AuthorId} (note {row.Rating:.0f}/5)"
                for row in top_evidence.itertuples()
            )

        ranked["ReviewEvidence"] = ranked["RecipeId"].map(evidence_by_recipe)
        return ranked.reset_index(drop=True), neighbors

    def evaluate_cold_start(
        self,
        n_users=50,
        n_likes=3,
        n_test=3,
        top_k=10,
        random_state=42,
    ):
        """
        Évalue le cold start sur des utilisateurs historiques simulés.

        Pour chaque utilisateur éligible :
        - sélectionne n_likes recettes positives comme seules préférences connues ;
        - garde n_test autres recettes positives cachées pour le test ;
        - exclut l'utilisateur évalué des voisins ;
        - compare le top-k recommandé aux recettes cachées.

        Retourne (résumé, résultats_par_utilisateur).
        """
        if n_likes < 1 or n_test < 1 or top_k < 1 or n_users < 1:
            raise ValueError("n_users, n_likes, n_test et top_k doivent être supérieurs à zéro.")

        rng = np.random.default_rng(random_state)
        positive = self.interactions[self.interactions["Rating"] >= 4].copy()
        positive_groups = {
            str(uid): group["RecipeId"].astype(str).drop_duplicates().tolist()
            for uid, group in positive.groupby("AuthorId")
        }

        eligible_ids = [
            uid for uid, ids in positive_groups.items()
            if len(ids) >= n_likes + n_test
        ]

        if not eligible_ids:
            raise ValueError(
                "Pas assez d'utilisateurs avec suffisamment de recettes positives "
                "pour séparer les choix connus et les recettes de test."
            )

        rng.shuffle(eligible_ids)
        eligible_ids = eligible_ids[:min(n_users, len(eligible_ids))]

        user_results = []
        print(
            f"Évaluation cold start : {len(eligible_ids)} utilisateurs, "
            f"{n_likes} recettes connues, {n_test} recettes cachées.",
            flush=True,
        )

        for index, author_id in enumerate(eligible_ids, start=1):
            positive_ids = positive_groups[author_id]
            sampled = rng.choice(
                positive_ids,
                size=n_likes + n_test,
                replace=False,
            )
            liked_ids = [str(rid) for rid in sampled[:n_likes]]
            hidden_ids = set(str(rid) for rid in sampled[n_likes:])

            recommendations, neighbors = self.recommend_from_likes(
                liked_recipe_ids=liked_ids,
                top_n=top_k,
                exclude_author_id=author_id,
            )

            recommended_ids = (
                recommendations["RecipeId"].astype(str).tolist()
                if not recommendations.empty
                else []
            )
            hits = len(set(recommended_ids) & hidden_ids)

            # Convention standard Precision@K : hits / K.
            precision = hits / top_k if top_k > 0 else 0.0
            recall = hits / len(hidden_ids) if hidden_ids else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            user_results.append({
                "AuthorId": author_id,
                "N_Likes": len(liked_ids),
                "N_Hidden": len(hidden_ids),
                "N_Recommendations": len(recommended_ids),
                "Hits": hits,
                f"Precision@{top_k}": precision,
                f"Recall@{top_k}": recall,
                f"F1@{top_k}": f1,
                "SimilarUsers": len(neighbors),
            })

            if index % 10 == 0 or index == len(eligible_ids):
                print(f"  Progression : {index}/{len(eligible_ids)}", flush=True)

        results = pd.DataFrame(user_results)
        metric_columns = [
            f"Precision@{top_k}",
            f"Recall@{top_k}",
            f"F1@{top_k}",
        ]
        summary = pd.DataFrame({
            "Metric": metric_columns,
            "Mean": [results[col].mean() for col in metric_columns],
            "Median": [results[col].median() for col in metric_columns],
        })

        print("\nRésultats moyens de l'évaluation cold start :")
        print(summary.to_string(index=False))
        return summary, results
