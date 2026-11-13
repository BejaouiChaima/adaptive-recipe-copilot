
from pathlib import Path
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

OUTPUT_PATH = ROOT / "src" / "evaluation" / "query_feasibility.csv"

print("Chargement des données...")

recipes = pd.read_csv(RECIPES_PATH)

with open(QUERIES_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)


# ============================================================
# 2. PARSING DES INGREDIENTS
# ============================================================

def parse_ingredients(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    # Format R : c("ingredient1", "ingredient2")
    if value.startswith("c("):

        ingredients = re.findall(
            r'"((?:\\.|[^"\\])*)"',
            value
        )

        return [
            ingredient.strip()
            for ingredient in ingredients
            if ingredient.strip()
        ]

    # Autres formats : liste Python
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


# ============================================================
# 3. NORMALISATION DES INGREDIENTS
# ============================================================

PROTECTED_WORDS = {
    "asparagus",
    "couscous",
    "hummus",
    "molasses",
    "citrus"
}


def normalize_text(text):

    text = str(text).lower().strip()

    # Suppression des accents
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Uniformisation de la ponctuation
    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def singularize_word(word):

    if word in PROTECTED_WORDS:
        return word

    # berries -> berry
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # tomatoes -> tomato
    # potatoes -> potato
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]

    # dishes -> dish
    # peaches -> peach
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]

    # eggs -> egg
    # onions -> onion
    # bananas -> banana
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def normalize_ingredient(ingredient):

    text = normalize_text(ingredient)

    if not text:
        return ""

    words = text.split()

    # On normalise uniquement les pluriels
    # de manière contrôlée.
    words = [
        singularize_word(word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# 4. COMPARAISON DES INGREDIENTS
# ============================================================

def is_wanted_ingredient_present(wanted, recipe_ingredients):

    wanted_normalized = normalize_ingredient(wanted)

    for ingredient in recipe_ingredients:

        ingredient_normalized = normalize_ingredient(ingredient)

        # Correspondance exacte après normalisation
        if ingredient_normalized == wanted_normalized:
            return True

    return False


def is_avoided_ingredient_present(avoided, recipe_ingredients):

    avoided_normalized = normalize_ingredient(avoided)

    for ingredient in recipe_ingredients:

        ingredient_normalized = normalize_ingredient(ingredient)

        words = ingredient_normalized.split()

        # Exceptions : les laits végétaux ne sont pas du lait animal.
        if avoided_normalized == "milk":

            plant_milks = {
                "coconut milk",
                "almond milk",
                "soy milk",
                "oat milk",
                "rice milk"
            }

            if ingredient_normalized in plant_milks:
                continue

        # Exceptions : les beurres végétaux ne sont pas du beurre laitier.
        if avoided_normalized == "butter":

            plant_butters = {
                "peanut butter",
                "almond butter",
                "cashew butter"
            }

            if ingredient_normalized in plant_butters:
                continue

        # Correspondance exacte ou présence comme mot entier.
        if (
            ingredient_normalized == avoided_normalized
            or avoided_normalized in words
        ):
            return True

    return False


# ============================================================
# 5. PARSING DU TEMPS
# ============================================================

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
# 6. PREPARATION DES RECETTES
# ============================================================

print("Parsing des ingrédients...")

recipes["ingredients_parsed"] = (
    recipes["RecipeIngredientParts"].apply(parse_ingredients)
)

print("Conversion du temps...")

recipes["total_time_minutes"] = (
    recipes["TotalTime"].apply(parse_time)
)

recipes["Calories"] = pd.to_numeric(
    recipes["Calories"],
    errors="coerce"
)

# On prépare une seule fois les recettes
# ayant les informations nécessaires.

valid_recipes = recipes.dropna(
    subset=["total_time_minutes", "Calories"]
).copy()

print("\nVERIFICATION DU PARSING")

for i in range(min(5, len(recipes))):

    print("\nValeur originale :")
    print(recipes["RecipeIngredientParts"].iloc[i])

    print("Valeur parsée :")
    print(recipes["ingredients_parsed"].iloc[i])


# ============================================================
# 7. EVALUATION DE FAISABILITE DES REQUETES
# ============================================================

results = []

print("\nEvaluation des requêtes...")

for query in queries:

    query_id = query["id"]

    wanted = query["wanted_ingredients"]
    avoided = query["avoided_ingredients"]

    max_time = query["max_time"]
    max_calories = query["max_calories"]

    candidates = valid_recipes[
        (valid_recipes["total_time_minutes"] <= max_time)
        &
        (valid_recipes["Calories"] <= max_calories)
    ].copy()

    # --------------------------------------------------------
    # Ingrédients interdits
    # --------------------------------------------------------

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

    total_candidates = len(candidates)

    # --------------------------------------------------------
    # Couverture des ingrédients souhaités
    # --------------------------------------------------------

    def calculate_coverage(ingredients):

        if not wanted:
            return 0

        matched = sum(
            is_wanted_ingredient_present(
                ingredient,
                ingredients
            )
            for ingredient in wanted
        )

        return matched / len(wanted)

    candidates["ingredient_coverage"] = (
        candidates["ingredients_parsed"].apply(
            calculate_coverage
        )
    )

    relevant_candidates = candidates[
        candidates["ingredient_coverage"] > 0
    ]

    all_wanted_candidates = candidates[
        candidates["ingredient_coverage"] == 1
    ]

    results.append({

        "query_id": query_id,

        "total_candidates": total_candidates,

        "recipes_with_wanted_ingredients": len(
            relevant_candidates
        ),

        "recipes_with_all_wanted_ingredients": len(
            all_wanted_candidates
        ),

        "average_ingredient_coverage": (
            relevant_candidates["ingredient_coverage"].mean()
            if len(relevant_candidates) > 0
            else 0
        )

    })

    print(
        f"{query_id} terminée : "
        f"{total_candidates} candidates"
    )


# ============================================================
# 8. EXPORT DES RESULTATS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 65)
print("RESULTATS DE FAISABILITE")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nFichier sauvegardé :")
print(OUTPUT_PATH)

print("\nNombre de requêtes :", len(results_df))

print("\nEvaluation terminée.")
