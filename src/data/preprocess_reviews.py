
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"
RECIPES_PATH = PROJECT_ROOT / "data" / "raw" / "recipes.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "reviews_clean.csv"


def preprocess_reviews():
    # 1. Charger les avis
    df = pd.read_csv(RAW_PATH)

    # 2. Nettoyer les espaces dans les colonnes textuelles
    for col in ["AuthorName", "Review"]:
        df[col] = df[col].map(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # 3. Remplacer les textes d'avis manquants par une chaîne vide
    df["Review"] = df["Review"].fillna("")

    # 4. Vérifier les identifiants des avis
    if df["ReviewId"].duplicated().any():
        raise ValueError("Des ReviewId sont dupliqués.")

    # 5. Vérifier les notes
    invalid_ratings = ~df["Rating"].between(0, 5)

    if invalid_ratings.any():
        print("Notes en dehors de l'intervalle 0-5 :",
              int(invalid_ratings.sum()))

    # 6. Convertir les dates ; les dates invalides deviennent NaT
    for col in ["DateSubmitted", "DateModified"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # 7. Créer le dossier de sortie et enregistrer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Prétraitement des avis terminé.")
    print("Nombre d'avis :", len(df))
    print("Nombre de ReviewId uniques :", df["ReviewId"].nunique())
    print("Avis sans texte :", int(df["Review"].eq("").sum()))
    print("Valeurs manquantes restantes :",
          int(df.isna().sum().sum()))
    print("Fichier enregistré :", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_reviews()
