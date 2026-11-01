
import pandas as pd
from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "recipes.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "recipes_clean.csv"


def preprocess_recipes():
    # 1. Charger les données brutes
    df = pd.read_csv(RAW_PATH)

    # 2. Supprimer les espaces inutiles dans les textes
    text_columns = df.select_dtypes(include=["object"]).columns

    for col in text_columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # 3. Traiter les valeurs manquantes des colonnes textuelles
    df["Description"] = df["Description"].fillna("")
    df["Keywords"] = df["Keywords"].fillna("")
    df["RecipeCategory"] = df["RecipeCategory"].fillna("Non renseignée")

    # 4. Vérifier l'unicité des identifiants
    if df["RecipeId"].duplicated().any():
        raise ValueError(
            "Des RecipeId sont dupliqués. Vérifie les doublons avant de continuer."
        )

    # 5. Ne pas supprimer les recettes ni imputer les valeurs
    # numériques manquantes automatiquement.

    # 6. Créer le dossier de sortie si nécessaire
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Enregistrer les données prétraitées
    df.to_csv(OUTPUT_PATH, index=False)

    print("Prétraitement terminé.")
    print("Nombre de recettes :", len(df))
    print("Fichier enregistré :", OUTPUT_PATH)
    print("Valeurs manquantes restantes :", int(df.isna().sum().sum()))


if __name__ == "__main__":
    preprocess_recipes()
