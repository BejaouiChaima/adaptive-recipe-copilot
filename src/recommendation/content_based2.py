
import re
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recipes_clean.csv"
)

TOP_K = 10
N_EVAL_QUERIES = 200
RANDOM_STATE = 42

# Les colonnes nécessaires uniquement.
USE_COLS = [
    "RecipeId",
    "Name",
    "RecipeIngredientParts",
    "RecipeCategory",
    "Description",
]


# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

def load_recipes():
    df = pd.read_csv(
        DATA_PATH,
        usecols=USE_COLS,
        low_memory=False,
    )

    df["RecipeId"] = pd.to_numeric(
        df["RecipeId"], errors="coerce"
    )

    df = df.dropna(subset=["RecipeId", "Name"]).copy()
    df["RecipeId"] = df["RecipeId"].astype("int64")

    # Une seule recette par identifiant.
    df = df.drop_duplicates(
        subset="RecipeId", keep="first"
    ).reset_index(drop=True)

    for col in [
        "Name",
        "RecipeIngredientParts",
        "RecipeCategory",
        "Description",
    ]:
        df[col] = df[col].fillna("").astype(str)

    # Les catégories manquantes ne doivent pas être
    # considérées comme une vraie catégorie commune.
    df["RecipeCategory"] = df["RecipeCategory"].str.strip()

    df["CategoryForEval"] = (
        df["RecipeCategory"]
        .str.lower()
        .str.strip()
    )

    df.loc[
        df["CategoryForEval"].isin(
            ["", "nan", "none", "non renseignée"]
        ),
        "CategoryForEval",
    ] = ""

    print("Dataset chargé.")
    print("Nombre de recettes :", len(df))
    print(
        "Nombre de catégories renseignées :",
        df["CategoryForEval"].nunique(),
    )

    return df


# ============================================================
# 3. CONSTRUCTION DU TEXTE
# ============================================================

def clean_text(value):
    """
    Nettoie le texte et les représentations telles que :
    c("chocolate", "sugar", "milk")
    """
    if not isinstance(value, str):
        return ""

    value = value.lower()

    # Supprimer la syntaxe fréquente des listes exportées.
    value = re.sub(r'\bc\s*\(', " ", value)
    value = re.sub(r'[\[\]{}()"\'`]', " ", value)

    # Conserver les lettres et les chiffres.
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def build_text(df, include_category=True):
    """
    Construit le texte utilisé par TF-IDF.

    Pendant l'évaluation, on exclut la catégorie
    pour pouvoir l'utiliser comme indicateur de pertinence.

    Pour le modèle final, on inclut la catégorie.
    """
    names = df["Name"].map(clean_text)
    ingredients = df["RecipeIngredientParts"].map(clean_text)
    descriptions = df["Description"].map(clean_text)

    if include_category:
        categories = df["RecipeCategory"].map(clean_text)

        # Répéter la catégorie lui donne davantage de poids
        # dans le texte final.
        text = (
            names + " "
            + ingredients + " "
            + ingredients + " "
            + categories + " "
            + categories + " "
            + descriptions
        )
    else:
        text = (
            names + " "
            + ingredients + " "
            + ingredients + " "
            + descriptions
        )

    # Eviter les documents totalement vides.
    text = text.replace("", "recipe")

    return text


# ============================================================
# 4. METRIQUES D'EVALUATION
# ============================================================

def precision_at_k(relevant, k):
    relevant = np.asarray(relevant[:k], dtype=int)
    return relevant.sum() / k


def recall_at_k(relevant, total_relevant, k):
    if total_relevant == 0:
        return np.nan

    return np.sum(relevant[:k]) / total_relevant


def average_precision_at_k(relevant, k):
    relevant = np.asarray(relevant[:k], dtype=int)

    if relevant.sum() == 0:
        return 0.0

    precisions = [
        relevant[i] * relevant[: i + 1].sum() / (i + 1)
        for i in range(len(relevant))
    ]

    return np.sum(precisions) / min(
        int(relevant.sum()), k
    )


def ndcg_at_k(relevant, k):
    relevant = np.asarray(relevant[:k], dtype=float)

    discounts = np.log2(
        np.arange(2, len(relevant) + 2)
    )

    dcg = np.sum(relevant / discounts)

    ideal_length = min(int(relevant.sum()), k)

    if ideal_length == 0:
        return 0.0

    ideal = np.sum(
        np.ones(ideal_length)
        / np.log2(np.arange(2, ideal_length + 2))
    )

    return dcg / ideal


# ============================================================
# 5. EVALUATION SUR UN JEU DE TEST
# ============================================================

def evaluate_model(df):
    """
    Evaluation proxy :
    une recette est considérée comme pertinente si elle
    partage la même catégorie renseignée que la requête.

    La catégorie est exclue des caractéristiques TF-IDF
    pendant cette évaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # On évalue un échantillon de requêtes pour limiter
    # le temps de calcul.
    if len(test_df) > N_EVAL_QUERIES:
        test_df = test_df.sample(
            n=N_EVAL_QUERIES,
            random_state=RANDOM_STATE,
        )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_text = build_text(
        train_df, include_category=False
    )
    test_text = build_text(
        test_df, include_category=False
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    print("\n--- Evaluation du modèle ---")
    print("Recettes candidates :", len(train_df))
    print("Requêtes de test :", len(test_df))
    print("Nombre de caractéristiques :", X_train.shape[1])

    metric_results = {
        "Precision@10": [],
        "Recall@10": [],
        "MAP@10": [],
        "NDCG@10": [],
    }

    valid_queries = 0

    train_categories = train_df[
        "CategoryForEval"
    ].to_numpy()

    for i in range(len(test_df)):
        query_category = test_df.loc[
            i, "CategoryForEval"
        ]

        # Sans catégorie connue, impossible d'utiliser
        # cet indicateur de pertinence.
        if not query_category:
            continue

        relevant_mask = (
            train_categories == query_category
        )

        total_relevant = int(relevant_mask.sum())

        if total_relevant == 0:
            continue

        # TF-IDF est normalisé par défaut :
        # le produit scalaire correspond à la similarité cosinus.
        scores = (
            X_train @ X_test[i].T
        ).toarray().ravel()

        # Sélection des 10 meilleurs candidats.
        top_indices = np.argpartition(
            scores, -min(TOP_K, len(scores))
        )[-min(TOP_K, len(scores)):]

        top_indices = top_indices[
            np.argsort(scores[top_indices])[::-1]
        ]

        relevant_at_k = relevant_mask[top_indices].astype(int)

        metric_results["Precision@10"].append(
            precision_at_k(relevant_at_k, TOP_K)
        )

        metric_results["Recall@10"].append(
            recall_at_k(
                relevant_at_k, total_relevant, TOP_K
            )
        )

        metric_results["MAP@10"].append(
            average_precision_at_k(
                relevant_at_k, TOP_K
            )
        )

        metric_results["NDCG@10"].append(
            ndcg_at_k(relevant_at_k, TOP_K)
        )

        valid_queries += 1

    print("Requêtes évaluables :", valid_queries)

    if valid_queries == 0:
        print(
            "Evaluation impossible : aucune requête "
            "ne possède une catégorie exploitable."
        )
        return

    print("\nRésultats moyens :")

    for metric, values in metric_results.items():
        print(f"{metric}: {np.mean(values):.4f}")

    print(
        "\nAttention : ces métriques utilisent la catégorie "
        "comme indicateur approximatif de pertinence."
    )


# ============================================================
# 6. ENTRAINEMENT DU MODELE FINAL
# ============================================================

def train_final_model(df):
    """
    Ajuste TF-IDF sur toutes les recettes.
    Aucun calcul de matrice dense N x N n'est effectué.
    """
    texts = build_text(df, include_category=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        dtype=np.float32,
    )

    recipe_matrix = vectorizer.fit_transform(texts)

    print("\n--- Modèle final entraîné ---")
    print("Nombre de recettes :", recipe_matrix.shape[0])
    print("Nombre de caractéristiques :", recipe_matrix.shape[1])
    print("Matrice TF-IDF :", recipe_matrix.shape)

    return vectorizer, recipe_matrix


# ============================================================
# 7. RECOMMANDATION DE RECETTES SIMILAIRES
# ============================================================

def recommend_by_name(
    query,
    df,
    vectorizer,
    recipe_matrix,
    top_n=10,
):
    """
    Recherche une recette par nom puis recommande
    les recettes les plus similaires.
    """
    matches = df[
        df["Name"].str.contains(
            query, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        print("Aucune recette trouvée pour :", query)
        return pd.DataFrame()

    recipe = matches.iloc[0]
    recipe_index = recipe.name

    print("\nRecette recherchée :", recipe["Name"])
    print("Catégorie :", recipe["RecipeCategory"])

    if len(matches) > 1:
        print(
            f"{len(matches)} recettes correspondent au texte. "
            "La première correspondance est utilisée."
        )

    query_vector = recipe_matrix[recipe_index]

    scores = (
        recipe_matrix @ query_vector.T
    ).toarray().ravel()

    # Ne pas recommander la recette elle-même.
    scores[recipe_index] = -1.0

    top_indices = np.argpartition(
        scores, -min(top_n, len(scores) - 1)
    )[-min(top_n, len(scores) - 1):]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    results = df.iloc[top_indices][
        ["RecipeId", "Name", "RecipeCategory",
         "RecipeIngredientParts"]
    ].copy()

    results["SimilarityScore"] = scores[top_indices]

    # Pour faciliter la lecture.
    results["SimilarityScore"] = (
        results["SimilarityScore"].round(4)
    )

    return results.reset_index(drop=True)


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    recipes = load_recipes()

    # Evaluation avant le modèle final.
    evaluate_model(recipes)

    # Modèle final : toutes les recettes.
    vectorizer, recipe_matrix = train_final_model(recipes)

    # Test concret.
    recommendations = recommend_by_name(
        "Oatmeal Cookies",
        recipes,
        vectorizer,
        recipe_matrix,
        top_n=10,
    )

    print("\n--- Top 10 des recettes similaires ---")

    if not recommendations.empty:
        print(
            recommendations.to_string(index=False)
        )

