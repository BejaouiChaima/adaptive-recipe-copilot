
import ast
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PREPARATION DES DONNEES
# --------------------------------------------------

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = re.sub(r"[\[\]{}()\"'`]", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_ingredients(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple)):
            return [
                str(item).strip().lower()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    value = str(value).strip("[]")
    ingredients = re.split(r",|;|\|", value)

    return [
        item.strip(" '\"\t").lower()
        for item in ingredients
        if item.strip(" '\"\t")
    ]


def parse_time(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    # Format ISO 8601 : PT1H30M
    match = re.match(
        r"^PT(?:(\d+(?:\.\d+)?)H)?"
        r"(?:(\d+(?:\.\d+)?)M)?"
        r"(?:(\d+(?:\.\d+)?)S)?$",
        value
    )

    if match:
        hours, minutes, seconds = [
            float(x or 0) for x in match.groups()
        ]

        return hours * 60 + minutes + seconds / 60

    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_calories(row):
    # Cas où Calories existe déjà.
    if "Calories" in row.index:
        try:
            value = float(row["Calories"])
            if np.isfinite(value):
                return value
        except (ValueError, TypeError):
            pass

    # Food.com : première valeur de Nutrition = calories.
    if "Nutrition" in row.index:
        try:
            nutrition = ast.literal_eval(str(row["Nutrition"]))

            if isinstance(nutrition, (list, tuple)):
                return float(nutrition[0])

        except (ValueError, SyntaxError, TypeError, IndexError):
            pass

    return np.nan


def prepare_recipes(df):

    required_columns = [
        "RecipeId",
        "Name",
        "RecipeIngredientParts",
        "RecipeCategory"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    df = df.copy()

    df["RecipeId"] = pd.to_numeric(
        df["RecipeId"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["RecipeId", "Name"]
    ).copy()

    df["RecipeId"] = df["RecipeId"].astype(int)

    for col in [
        "Name",
        "RecipeIngredientParts",
        "RecipeCategory",
        "Description"
    ]:

        if col not in df.columns:
            df[col] = ""

        df[col] = df[col].fillna("").astype(str)

    # Temps de préparation total.
    if "TotalTime" in df.columns:
        df["TimeMinutes"] = df["TotalTime"].apply(parse_time)

    elif "PrepTime" in df.columns:
        df["TimeMinutes"] = df["PrepTime"].apply(parse_time)

    else:
        df["TimeMinutes"] = np.nan

    # Calories.
    df["CaloriesValue"] = df.apply(
        parse_calories,
        axis=1
    )

    # Liste d'ingrédients.
    df["IngredientsList"] = (
        df["RecipeIngredientParts"].apply(parse_ingredients)
    )

    df["CategoryClean"] = (
        df["RecipeCategory"].map(clean_text)
    )

    df = df.drop_duplicates(
        subset="RecipeId"
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# CONSTRUCTION DU TEXTE
# --------------------------------------------------

def build_recipe_text(df):

    names = df["Name"].map(clean_text)

    ingredients = (
        df["RecipeIngredientParts"].map(clean_text)
    )

    categories = (
        df["RecipeCategory"].map(clean_text)
    )

    descriptions = (
        df["Description"].map(clean_text)
    )

    # Les ingrédients sont répétés pour leur donner
    # plus d'importance dans la représentation TF-IDF.

    text = (
        names + " "
        + ingredients + " "
        + ingredients + " "
        + categories + " "
        + descriptions
    )

    return text


# --------------------------------------------------
# MODELE CONTENT-BASED
# --------------------------------------------------

class ContentBasedRecommender:

    def __init__(
        self,
        vectorizer,
        recipe_matrix,
        recipes
    ):

        self.vectorizer = vectorizer
        self.recipe_matrix = recipe_matrix
        self.recipes = recipes.reset_index(drop=True)

    def recommend(
        self,
        wanted_ingredients=None,
        avoided_ingredients=None,
        max_calories=None,
        max_time=None,
        category=None,
        top_n=10
    ):

        wanted = wanted_ingredients or []
        avoided = avoided_ingredients or []

        wanted = [
            clean_text(x)
            for x in wanted
            if clean_text(x)
        ]

        avoided = [
            clean_text(x)
            for x in avoided
            if clean_text(x)
        ]

        candidates = self.recipes.copy()

        # ------------------------------------------
        # 1. FILTRE CALORIES
        # ------------------------------------------

        if max_calories is not None:

            candidates = candidates[
                candidates["CaloriesValue"].notna()
                &
                (
                    candidates["CaloriesValue"]
                    <= float(max_calories)
                )
            ]

        # ------------------------------------------
        # 2. FILTRE TEMPS
        # ------------------------------------------

        if max_time is not None:

            candidates = candidates[
                candidates["TimeMinutes"].notna()
                &
                (
                    candidates["TimeMinutes"]
                    <= float(max_time)
                )
            ]

        # ------------------------------------------
        # 3. FILTRE CATEGORIE
        # ------------------------------------------

        if category:

            category_clean = clean_text(category)

            candidates = candidates[
                candidates["CategoryClean"].str.contains(
                    category_clean,
                    regex=False,
                    na=False
                )
            ]

        # ------------------------------------------
        # 4. EXCLURE LES INGREDIENTS INTERDITS
        # ------------------------------------------

        if avoided:

            ingredient_text = (
                candidates["RecipeIngredientParts"]
                .map(clean_text)
            )

            for ingredient in avoided:

                candidates = candidates[
                    ~ingredient_text.loc[
                        candidates.index
                    ].str.contains(
                        re.escape(ingredient),
                        regex=True,
                        na=False
                    )
                ]

        if candidates.empty:
            return pd.DataFrame()

        # ------------------------------------------
        # 5. PROFIL UTILISATEUR
        # ------------------------------------------

        if not wanted and not category:

            raise ValueError(
                "Veuillez saisir au moins un ingrédient "
                "souhaité ou une catégorie."
            )

        query_text = " ".join(wanted)

        if category:
            query_text += " " + clean_text(category)

        query_vector = self.vectorizer.transform(
            [query_text]
        )

        # ------------------------------------------
        # 6. SIMILARITE COSINUS
        # ------------------------------------------

        candidate_indices = candidates.index.to_numpy()

        scores = cosine_similarity(
            self.recipe_matrix[candidate_indices],
            query_vector
        ).ravel()

        candidates["SimilarityScore"] = scores

        # ------------------------------------------
        # 7. BONUS INGREDIENTS SOUHAITES
        # ------------------------------------------

        if wanted:

            ingredient_text = (
                candidates["RecipeIngredientParts"]
                .map(clean_text)
            )

            coverage = np.zeros(
                len(candidates),
                dtype=np.float32
            )

            for ingredient in wanted:

                coverage += ingredient_text.str.contains(
                    re.escape(ingredient),
                    regex=True,
                    na=False
                ).to_numpy(dtype=np.float32)

            coverage /= len(wanted)

        else:

            coverage = np.zeros(
                len(candidates),
                dtype=np.float32
            )

        candidates["IngredientCoverage"] = coverage

        candidates["FinalScore"] = (
            0.8 * candidates["SimilarityScore"]
            + 0.2 * candidates["IngredientCoverage"]
        )

        # ------------------------------------------
        # 8. TOP N
        # ------------------------------------------

        columns = [
            "RecipeId",
            "Name",
            "RecipeCategory",
            "RecipeIngredientParts",
            "CaloriesValue",
            "TimeMinutes",
            "SimilarityScore",
            "IngredientCoverage",
            "FinalScore"
        ]

        return (
            candidates
            .sort_values(
                "FinalScore",
                ascending=False
            )
            [columns]
            .head(top_n)
            .reset_index(drop=True)
        )


# --------------------------------------------------
# ENTRAINEMENT TF-IDF
# --------------------------------------------------

def fit_recommender(recipes):

    texts = build_recipe_text(recipes)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        ngram_range=(1, 2),
        dtype=np.float32
    )

    recipe_matrix = vectorizer.fit_transform(texts)

    model = ContentBasedRecommender(
        vectorizer,
        recipe_matrix,
        recipes
    )

    return model
