from pathlib import Path
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

from src.recommendation.content_based2 import (
    prepare_recipes,
    fit_recommender,
    build_recipe_text
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "recipes_clean.csv"
MODEL_DIR = ROOT_DIR / "models"
TEST_DIR = ROOT_DIR / "tests"

MODEL_PATH = MODEL_DIR / "content_based_model.pkl"
EVAL_MODEL_PATH = MODEL_DIR / "content_based_eval.pkl"
TEST_RECIPES_PATH = TEST_DIR / "test_recipes_catalog.csv"


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement du dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = prepare_recipes(df)

    print("Nombre total de recettes :", len(df))

    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("Train :", len(train_df))
    print("Test :", len(test_df))

    print("\nEntraînement sur les 90 %...")

    eval_model = fit_recommender(train_df)

    # Conserver les 10 % de test comme catalogue complet.
    test_text = build_recipe_text(test_df)

    test_matrix = eval_model.vectorizer.transform(test_text)

    eval_model.recipes = test_df
    eval_model.recipe_matrix = test_matrix

    with open(EVAL_MODEL_PATH, "wb") as f:
        pickle.dump(eval_model, f)

    print("Modèle d'évaluation sauvegardé.")

    test_df[
        [
            "RecipeId",
            "Name",
            "RecipeCategory",
            "RecipeIngredientParts",
            "CaloriesValue",
            "TimeMinutes"
        ]
    ].to_csv(
        TEST_RECIPES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Catalogue test sauvegardé.")

    print("\nEntraînement final sur 100 % des recettes...")

    final_model = fit_recommender(df)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    print("Modèle final sauvegardé.")
    print("\nEntraînement terminé.")


if __name__ == "__main__":
    main()