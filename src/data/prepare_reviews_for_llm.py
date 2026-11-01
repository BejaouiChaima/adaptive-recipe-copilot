"""
Prépare un fichier CSV léger destiné à la classification de sentiment par un LLM.

Entrée  : data/processed/reviews_clean.csv
Sortie  : data/processed/reviews_for_llm.csv

Le fichier de sortie ne contient que 3 colonnes :
    - ReviewId : identifiant de l'avis (clé de jointure)
    - Review   : texte de l'avis, nettoyé
    - Rating   : note 0-5, utilisée comme indice par le LLM

Aucune ligne n'est supprimée : la ligne N du fichier de sortie correspond
exactement à la ligne N de reviews_clean.csv. Cela permet de reprendre le
traitement LLM à une ligne précise (--start-line) sans décalage.

Usage :
    python src/data/prepare_reviews_for_llm.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "reviews_for_llm.csv"

COLUMNS = ["ReviewId", "Review", "Rating"]


def prepare_reviews_for_llm(input_path: Path = INPUT_PATH,
                            output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    # 1. Charger uniquement les colonnes utiles
    df = pd.read_csv(
        input_path,
        usecols=COLUMNS,
        dtype={"ReviewId": "Int64", "Review": "string", "Rating": "Int64"},
    )

    # 2. Nettoyer le texte : espaces multiples, retours à la ligne, NaN
    df["Review"] = (
        df["Review"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 3. Normaliser la note (entier 0-5, NaN conservé)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").round().astype("Int64")

    # 4. Enregistrer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # 5. Rapport
    print("Préparation terminée.")
    print("Lignes :", len(df))
    print("ReviewId uniques :", df["ReviewId"].nunique())
    print("Avis vides :", int(df["Review"].eq("").sum()))
    print("Avis sans note :", int(df["Rating"].isna().sum()))
    print("Longueur moyenne du texte :", round(df["Review"].str.len().mean(), 1),
          "caractères")
    print("Fichier enregistré :", output_path)
    return df


if __name__ == "__main__":
    prepare_reviews_for_llm()
