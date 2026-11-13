
from pathlib import Path
from collections import Counter
import json
import re
import ast
import unicodedata

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RECIPES_PATH = ROOT / "data" / "processed" / "recipes_clean.csv"
QUERIES_PATH = ROOT / "src" / "evaluation" / "benchmark_queries.json"

OUTPUT_PATH = ROOT / "src" / "evaluation" / "ingredient_matching_audit.csv"


# ============================================================
# 2. FONCTIONS DE PARSING ET NORMALISATION
# ============================================================

def parse_ingredients(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if value.startswith("c("):

        ingredients = re.findall(
            r'"((?:\\.|[^"\\])*)"',
            value
        )

        return [
            x.strip()
            for x in ingredients
            if x.strip()
        ]

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple)):

            return [
                str(x).strip()
                for x in parsed
                if pd.notna(x)
            ]

    except (ValueError, SyntaxError):
        pass

    return []


PROTECTED_WORDS = {
    "asparagus",
    "couscous",
    "hummus",
    "molasses",
    "citrus"
}


def normalize_text(text):

    text = str(text).lower().strip()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def singularize_word(word):

    if word in PROTECTED_WORDS:
        return word

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("oes") and len(word) > 4:
        return word[:-2]

    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]

    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def normalize_ingredient(ingredient):

    words = normalize_text(ingredient).split()

    words = [
        singularize_word(word)
        for word in words
    ]

    return " ".join(words)


def is_avoided_ingredient_present(avoided, ingredients):

    avoided_normalized = normalize_ingredient(avoided)

    plant_milks = {
        "coconut milk",
        "almond milk",
        "soy milk",
        "oat milk",
        "rice milk"
    }

    plant_butters = {
        "peanut butter",
        "almond butter",
        "cashew butter"
    }

    for ingredient in ingredients:

        normalized = normalize_ingredient(ingredient)

        if avoided_normalized == "milk" and normalized in plant_milks:
            continue

        if avoided_normalized == "butter" and normalized in plant_butters:
            continue

        words = normalized.split()

        if (
            normalized == avoided_normalized
            or avoided_normalized in words
        ):
            return True

    return False


def parse_time(value):

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    match = re.fullmatch(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        value
    )

    if not match:
        return None

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 60 + minutes + seconds / 60


# ============================================================
# 3. CHARGEMENT DES DONNEES
# ============================================================

print("Chargement des recettes...")

recipes = pd.read_csv(RECIPES_PATH)

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)

recipes["ingredients_parsed"] = (
    recipes["RecipeIngredientParts"].apply(parse_ingredients)
)

recipes["total_time_minutes"] = (
    recipes["TotalTime"].apply(parse_time)
)

recipes["Calories"] = pd.to_numeric(
    recipes["Calories"],
    errors="coerce"
)

recipes = recipes.dropna(
    subset=["total_time_minutes", "Calories"]
).copy()


# ============================================================
# 4. AUDIT PAR REQUETE ET PAR INGREDIENT
# ============================================================

audit_results = []

for query in queries:

    query_id = query["id"]

    wanted = query["wanted_ingredients"]
    avoided = query["avoided_ingredients"]

    max_time = query["max_time"]
    max_calories = query["max_calories"]

    print(f"\nAnalyse de {query_id}...")

    # Même catalogue admissible que dans le script précédent.

    candidates = recipes[
        (recipes["total_time_minutes"] <= max_time)
        &
        (recipes["Calories"] <= max_calories)
    ].copy()

    if avoided:

        candidates = candidates[
            ~candidates["ingredients_parsed"].apply(
                lambda ingredients: any(
                    is_avoided_ingredient_present(
                        ingredient,
                        ingredients
                    )
                    for ingredient in avoided
                )
            )
        ]

    # Fréquence des noms d'ingrédients dans les recettes candidates.

    ingredient_counter = Counter()

    for ingredients in candidates["ingredients_parsed"]:

        for ingredient in ingredients:

            normalized = normalize_ingredient(ingredient)

            if normalized:
                ingredient_counter[normalized] += 1

    # Audit de chaque ingrédient souhaité.

    for wanted_ingredient in wanted:

        wanted_normalized = normalize_ingredient(
            wanted_ingredient
        )

        matched_recipes = candidates[
            candidates["ingredients_parsed"].apply(
                lambda ingredients: any(
                    normalize_ingredient(x) == wanted_normalized
                    for x in ingredients
                )
            )
        ]

        # Variantes partageant au moins un mot normalisé.
        wanted_words = set(wanted_normalized.split())

        possible_variants = []

        for dataset_ingredient, frequency in ingredient_counter.items():

            if dataset_ingredient == wanted_normalized:
                continue

            dataset_words = set(dataset_ingredient.split())

            if wanted_words.intersection(dataset_words):

                possible_variants.append(
                    f"{dataset_ingredient} ({frequency})"
                )

        possible_variants = sorted(
            possible_variants,
            key=lambda x: int(
                re.search(r"\((\d+)\)", x).group(1)
            ),
            reverse=True
        )

        # Trois exemples de recettes contenant l'ingrédient.

        examples = []

        for _, row in matched_recipes.head(3).iterrows():

            examples.append(
                f"{row['RecipeId']} - {row['Name']}"
            )

        audit_results.append({

            "query_id": query_id,

            "wanted_ingredient": wanted_ingredient,

            "normalized_ingredient": wanted_normalized,

            "matching_recipes": len(matched_recipes),

            "possible_dataset_variants": " | ".join(
                possible_variants[:15]
            ),

            "example_recipes": " | ".join(examples)

        })


# ============================================================
# 5. EXPORT
# ============================================================

audit_df = pd.DataFrame(audit_results)

audit_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("AUDIT DU MATCHING DES INGREDIENTS")
print("=" * 70)

print(audit_df.to_string(index=False))

print("\nFichier sauvegardé :")
print(OUTPUT_PATH)

print("\nNombre d'ingrédients analysés :", len(audit_df))

print("\nAudit terminé.")
