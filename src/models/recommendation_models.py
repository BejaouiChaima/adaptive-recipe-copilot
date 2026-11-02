
from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_list(value):
    """Convert a list-like recipe field into searchable text."""

    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return " ".join(str(item) for item in parsed)

    except (ValueError, SyntaxError):
        pass

    # Handle strings such as c("flour", "sugar")
    value = re.sub(r"^c\(", "", value)
    value = re.sub(r"\)$", "", value)
    value = value.replace('"', " ").replace("'", " ")
    value = value.replace(",", " ")

    return value


def get_top_indices(scores, top_n):
    """Return indices of the highest scores without sorting everything."""

    if len(scores) == 0 or top_n <= 0:
        return np.array([], dtype=int)

    top_n = min(top_n, len(scores))

    if top_n == len(scores):
        indices = np.arange(len(scores))
    else:
        indices = np.argpartition(
            scores, -top_n
        )[-top_n:]

    # Sort selected indices by descending score
    indices = indices[
        np.argsort(scores[indices])[::-1]
    ]

    return indices


# ============================================================
# MODEL 1: CONTENT-BASED FILTERING (TF-IDF)
# ============================================================

class ContentBasedModel:

    def __init__(self, max_features=100000):

        self.max_features = max_features

        self.recipes = None
        self.vectorizer = None
        self.recipe_matrix = None

        self.recipe_to_index = {}
        self.index_to_recipe = {}

    def fit(self, recipes):

        print("\nTraining Content-Based TF-IDF...")

        required_column = "RecipeId"

        if required_column not in recipes.columns:
            raise ValueError("Missing column: RecipeId")

        self.recipes = recipes.copy()

        self.recipes["RecipeId"] = (
            self.recipes["RecipeId"].astype(str)
        )

        self.recipes = self.recipes.drop_duplicates(
            subset=["RecipeId"]
        ).reset_index(drop=True)

        text_columns = [
            "Name",
            "RecipeCategory",
            "Keywords",
            "Description",
            "RecipeIngredientParts"
        ]

        # Create missing text columns if necessary
        for column in text_columns:
            if column not in self.recipes.columns:
                self.recipes[column] = ""

        # Prepare text without iterrows()
        documents = []

        for column in text_columns:

            if column in [
                "Keywords",
                "RecipeIngredientParts"
            ]:
                column_text = self.recipes[column].map(
                    parse_list
                )
            else:
                column_text = (
                    self.recipes[column]
                    .fillna("")
                    .astype(str)
                )

            documents.append(column_text)

        combined_text = documents[0]

        for column_text in documents[1:]:
            combined_text = (
                combined_text + " " + column_text
            )

        combined_text = combined_text.fillna("")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            dtype=np.float32,
            sublinear_tf=True
        )

        self.recipe_matrix = self.vectorizer.fit_transform(
            combined_text
        )

        # Recipe mappings
        self.recipe_to_index = {
            recipe_id: index
            for index, recipe_id in enumerate(
                self.recipes["RecipeId"]
            )
        }

        self.index_to_recipe = {
            index: recipe_id
            for recipe_id, index in self.recipe_to_index.items()
        }

        print("Content-Based training completed.")
        print("Recipes:", len(self.recipes))
        print("TF-IDF matrix:", self.recipe_matrix.shape)

        return self

    def recommend(self, liked_recipe_ids, top_n=10):

        liked_indices = []

        for recipe_id in liked_recipe_ids:

            recipe_id = str(recipe_id)

            if recipe_id in self.recipe_to_index:
                liked_indices.append(
                    self.recipe_to_index[recipe_id]
                )

        # Remove duplicate indices
        liked_indices = list(set(liked_indices))

        if not liked_indices:
            return pd.DataFrame(
                columns=["RecipeId", "Score"]
            )

        # Average profile of liked recipes
        user_profile = self.recipe_matrix[
            liked_indices
        ].mean(axis=0)

        # Similarity with all recipes
        scores = np.asarray(
            self.recipe_matrix @ user_profile.T
        ).ravel()

        # Exclude already liked recipes
        scores[liked_indices] = -np.inf

        top_indices = get_top_indices(scores, top_n)

        return pd.DataFrame({
            "RecipeId": [
                self.index_to_recipe[index]
                for index in top_indices
            ],
            "Score": scores[top_indices]
        })


# ============================================================
# MODEL 2: COLLABORATIVE FILTERING (SVD)
# ============================================================

class SVDRecommender:

    def __init__(self, n_components=50):

        self.n_components = n_components

        self.svd = None
        self.item_factors = None

        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_item = {}

        self.global_mean = 0.0

    def fit(self, train_reviews):

        print("\nTraining Collaborative Filtering SVD...")

        required_columns = [
            "AuthorId",
            "RecipeId",
            "Rating"
        ]

        for column in required_columns:
            if column not in train_reviews.columns:
                raise ValueError(
                    f"Missing column: {column}"
                )

        data = train_reviews.copy()

        data["AuthorId"] = data["AuthorId"].astype(str)
        data["RecipeId"] = data["RecipeId"].astype(str)

        data["Rating"] = pd.to_numeric(
            data["Rating"], errors="coerce"
        )

        data = data.dropna(
            subset=required_columns
        )

        # Aggregate duplicate interactions
        data = data.groupby(
            ["AuthorId", "RecipeId"],
            as_index=False
        )["Rating"].mean()

        if data.empty:
            raise ValueError("Train dataset is empty.")

        users = data["AuthorId"].unique()
        items = data["RecipeId"].unique()

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(users)
        }

        self.item_to_index = {
            recipe_id: index
            for index, recipe_id in enumerate(items)
        }

        self.index_to_item = {
            index: recipe_id
            for recipe_id, index in self.item_to_index.items()
        }

        row_indices = data["AuthorId"].map(
            self.user_to_index
        ).to_numpy()

        col_indices = data["RecipeId"].map(
            self.item_to_index
        ).to_numpy()

        self.global_mean = float(data["Rating"].mean())

        # Center ratings around the global mean
        values = (
            data["Rating"].to_numpy(dtype=np.float32)
            - self.global_mean
        )

        rating_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(users), len(items)),
            dtype=np.float32
        )

        # Select valid number of components
        n_components = min(
            self.n_components,
            min(rating_matrix.shape) - 1
        )

        if n_components < 1:
            raise ValueError(
                "Not enough users or recipes for SVD."
            )

        self.svd = TruncatedSVD(
            n_components=n_components,
            n_iter=7,
            random_state=42
        )

        # Train using Train interactions only
        self.svd.fit(rating_matrix)

        # Latent factors for recipes
        self.item_factors = self.svd.components_.T

        print("SVD training completed.")
        print("Users:", len(users))
        print("Recipes:", len(items))
        print("Components:", n_components)
        print("Rating matrix:", rating_matrix.shape)

        return self

    def recommend(
        self,
        liked_recipe_ids,
        liked_ratings=None,
        top_n=10
    ):

        if liked_ratings is None:
            liked_ratings = [5.0] * len(liked_recipe_ids)

        if len(liked_recipe_ids) != len(liked_ratings):
            raise ValueError(
                "liked_recipe_ids and liked_ratings "
                "must have the same length."
            )

        liked_indices = []
        ratings = []

        for recipe_id, rating in zip(
            liked_recipe_ids,
            liked_ratings
        ):

            recipe_id = str(recipe_id)

            if recipe_id in self.item_to_index:

                liked_indices.append(
                    self.item_to_index[recipe_id]
                )

                ratings.append(float(rating))

        if not liked_indices:
            return pd.DataFrame(
                columns=["RecipeId", "Score"]
            )

        # Remove duplicate known recipes
        known_data = {}

        for index, rating in zip(
            liked_indices, ratings
        ):
            known_data[index] = rating

        liked_indices = list(known_data.keys())
        ratings = list(known_data.values())

        # Infer a new user profile from known likes
        factors = self.item_factors[liked_indices]

        centered_ratings = (
            np.asarray(ratings, dtype=np.float32)
            - self.global_mean
        )

        weights = np.maximum(
            centered_ratings,
            0.1
        )

        user_profile = np.average(
            factors,
            axis=0,
            weights=weights
        )

        # Predict preference scores
        scores = self.item_factors @ user_profile

        # Exclude known recipes
        scores[liked_indices] = -np.inf

        top_indices = get_top_indices(scores, top_n)

        return pd.DataFrame({
            "RecipeId": [
                self.index_to_item[index]
                for index in top_indices
            ],
            "Score": scores[top_indices]
        })


# ============================================================
# MODEL 3: HYBRID (TF-IDF + SVD)
# ============================================================

class HybridRecommender:

    def __init__(
        self,
        content_model,
        svd_model,
        content_weight=0.5,
        candidate_pool=500
    ):

        if not 0 <= content_weight <= 1:
            raise ValueError(
                "content_weight must be between 0 and 1."
            )

        self.content_model = content_model
        self.svd_model = svd_model

        self.content_weight = content_weight
        self.svd_weight = 1.0 - content_weight

        self.candidate_pool = candidate_pool

    def normalize_scores(self, scores):

        scores = np.asarray(
            scores,
            dtype=np.float32
        )

        if len(scores) == 0:
            return scores

        minimum = scores.min()
        maximum = scores.max()

        if maximum == minimum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def recommend(
        self,
        liked_recipe_ids,
        liked_ratings=None,
        top_n=10
    ):

        # Get candidates from Content-Based
        content_results = self.content_model.recommend(
            liked_recipe_ids,
            top_n=self.candidate_pool
        )

        # Get candidates from SVD
        svd_results = self.svd_model.recommend(
            liked_recipe_ids,
            liked_ratings=liked_ratings,
            top_n=self.candidate_pool
        )

        if content_results.empty and svd_results.empty:
            return pd.DataFrame(
                columns=["RecipeId", "Score"]
            )

        # Store original scores
        content_scores = dict(zip(
            content_results["RecipeId"],
            content_results["Score"]
        ))

        svd_scores = dict(zip(
            svd_results["RecipeId"],
            svd_results["Score"]
        ))

        # Union of candidates
        candidates = sorted(
            set(content_scores) | set(svd_scores)
        )

        # Normalize scores separately on each model's candidates
        content_ids = list(content_scores.keys())
        svd_ids = list(svd_scores.keys())

        normalized_content = self.normalize_scores(
            list(content_scores.values())
        )

        normalized_svd = self.normalize_scores(
            list(svd_scores.values())
        )

        content_normalized_dict = dict(
            zip(content_ids, normalized_content)
        )

        svd_normalized_dict = dict(
            zip(svd_ids, normalized_svd)
        )

        # Missing candidate score = 0
        final_scores = []

        for recipe_id in candidates:

            content_score = content_normalized_dict.get(
                recipe_id, 0.0
            )

            svd_score = svd_normalized_dict.get(
                recipe_id, 0.0
            )

            final_score = (
                self.content_weight * content_score
                + self.svd_weight * svd_score
            )

            final_scores.append(final_score)

        results = pd.DataFrame({
            "RecipeId": candidates,
            "Score": final_scores
        })

        # Exclude already liked recipes
        liked_set = set(map(str, liked_recipe_ids))

        results = results[
            ~results["RecipeId"].isin(liked_set)
        ]

        return results.sort_values(
            "Score",
            ascending=False
        ).head(top_n).reset_index(drop=True)
