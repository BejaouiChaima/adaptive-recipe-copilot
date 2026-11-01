
import ast
import pandas as pd

# Charger les recettes et les résultats du test
recipes = pd.read_csv("data/raw/recipes.csv", low_memory=False)
details = pd.read_csv("cold_start_svd_details.csv")

# Harmoniser les identifiants
recipes["RecipeId"] = recipes["RecipeId"].astype(str)

# Associer chaque identifiant au nom de sa recette
recipe_names = (
    recipes.drop_duplicates("RecipeId")
    .set_index("RecipeId")["Name"]
    .to_dict()
)

def parse_ids(value):
    """Convertit une liste enregistrée dans le CSV en liste Python."""
    if isinstance(value, list):
        return [str(x) for x in value]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    return []

def show_recipes(title, ids):
    print(f"\n{title}")
    if not ids:
        print("  Aucune recette")
        return

    for recipe_id in ids:
        name = recipe_names.get(recipe_id, "Nom introuvable")
        print(f"  ID {recipe_id} : {name}")

# Prendre un exemple du scénario 3 likes / 3 recettes cachées
sample = details[
    (details["N_Likes"] == 3) &
    (details["N_Test"] == 3)
].iloc[0]

liked = parse_ids(sample["LikedRecipeIds"])
hidden = parse_ids(sample["HiddenRecipeIds"])
content_recs = parse_ids(sample["ContentRecommendations"])
hybrid_recs = parse_ids(sample["HybridRecommendations"])

print("Utilisateur testé :", sample["AuthorId"])

show_recipes("RECETTES AIMÉES FOURNIES AU MODÈLE", liked)
show_recipes("RECETTES CACHÉES À RETROUVER", hidden)
show_recipes("TOP 10 CONTENT-BASED", content_recs)
show_recipes("TOP 10 HYBRIDE", hybrid_recs)

print("\nCorrespondances exactes :")
print("Content-Based :", len(set(content_recs) & set(hidden)))
print("Hybride       :", len(set(hybrid_recs) & set(hidden)))
