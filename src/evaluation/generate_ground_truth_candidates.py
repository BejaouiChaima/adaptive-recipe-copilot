import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "recipes_clean.csv"
QUERIES_PATH = ROOT_DIR / "src" / "evaluation" / "benchmark_queries.json"

OUTPUT_PATH = ROOT_DIR / "src" / "evaluation" / "ground_truth_candidates.csv"


# ============================================================
# 1. NORMALISATION ET PARSING
# ============================================================

def normalize(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_ingredients(value):
    """
    Convertit une chaîne de type :
    c("chicken", "rice", "salt")
    en liste Python.
    """

    if pd.isna(value):
        return []

    return [
        normalize(x)
        for x in re.findall(r'"([^"]*)"', str(value))
        if normalize(x)
    ]


def parse_time(value):
    """
    Convertit PT35M en 35 minutes.
    Convertit PT1H20M en 80 minutes.
    """

    if pd.isna(value):
        return None

    value = str(value)

    match = re.match(
        r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$",
        value
    )

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)

    return hours * 60 + minutes


# ============================================================
# 2. CORRESPONDANCE DES INGREDIENTS
# ============================================================

def ingredient_matches(recipe_ingredients, wanted):
    """
    Correspondance simple par sous-chaîne normalisée.
    On conserve ici une logique proche du diagnostic précédent.
    """

    wanted = normalize(wanted)

    return any(
        wanted in ingredient
        for ingredient in recipe_ingredients
    )


def contains_avoided(recipe_ingredients, avoided):
    for avoided_ingredient in avoided:
        if ingredient_matches(
            recipe_ingredients,
            avoided_ingredient
        ):
            return True

    return False


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

print("Chargement du dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"Nombre de recettes : {len(df)}")

print("Chargement des questions...")

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)


# ============================================================
# 4. PREPARATION DES RECETTES
# ============================================================

print("Préparation des ingrédients et des temps...")

df["parsed_ingredients"] = df["RecipeIngredientParts"].apply(
    parse_ingredients
)

df["time_minutes"] = df["TotalTime"].apply(parse_time)

df["Calories"] = pd.to_numeric(
    df["Calories"],
    errors="coerce"
)


# ============================================================
# 5. EXTRACTION DES CANDIDATS POUR CHAQUE QUESTION
# ============================================================

all_candidates = []

for query in queries:

    query_id = query["id"]

    wanted = query.get("wanted_ingredients", [])
    avoided = query.get("avoided_ingredients", [])

    max_time = query.get("max_time")
    max_calories = query.get("max_calories")

    print(f"\nTraitement {query_id}...")

    candidates = []

    for _, recipe in df.iterrows():

        ingredients = recipe["parsed_ingredients"]

        # Exclusion des ingrédients interdits
        if contains_avoided(ingredients, avoided):
            continue

        # Temps maximum
        if max_time is not None:
            recipe_time = recipe["time_minutes"]

            if pd.isna(recipe_time) or recipe_time > max_time:
                continue

        # Calories maximum
        if max_calories is not None:
            calories = recipe["Calories"]

            if pd.isna(calories) or calories > max_calories:
                continue

        # Ingrédients souhaités
        matched = [
            ingredient
            for ingredient in wanted
            if ingredient_matches(ingredients, ingredient)
        ]

        if len(matched) == 0:
            continue

        coverage = len(matched) / len(wanted)

        candidates.append({
            "query_id": query_id,
            "RecipeId": recipe["RecipeId"],
            "Name": recipe["Name"],
            "matched_ingredients": " | ".join(matched),
            "matched_count": len(matched),
            "wanted_count": len(wanted),
            "ingredient_coverage": round(coverage, 4),
            "TotalTime": recipe["TotalTime"],
            "Calories": recipe["Calories"],
            "RecipeIngredientParts": recipe["RecipeIngredientParts"],
            "relevance": ""
        })

    print(f"Recettes candidates : {len(candidates)}")

    all_candidates.extend(candidates)


# ============================================================
# 6. EXPORT
# ============================================================

results = pd.DataFrame(all_candidates)

results.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n======================================")
print("EXTRACTION TERMINEE")
print("======================================")

print(f"Total candidats : {len(results)}")
print(f"Fichier : {OUTPUT_PATH}")

print("\nNombre de candidats par question :")

if not results.empty:
    print(
        results.groupby("query_id")["RecipeId"]
        .count()
        .to_string()
    )