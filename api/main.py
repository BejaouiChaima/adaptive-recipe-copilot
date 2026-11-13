
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Important : permet à pickle de retrouver la classe du modèle
from src.recommendation.content_based2 import ContentBasedRecommender
from api.evaluation_logger import router as evaluation_router

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "content_based_model.pkl"

app = FastAPI(
    title="Adaptive Recipe Copilot API",
    version="1.0.0"
)
app.include_router(evaluation_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Le modèle est chargé une seule fois au démarrage
model = None


def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    print("Chargement du modèle Content-Based...")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Modèle chargé avec succès.")


@app.on_event("startup")
def startup_event():
    load_model()


class RecommendationRequest(BaseModel):
    wanted_ingredients: list[str] = Field(default_factory=list)
    avoided_ingredients: list[str] = Field(default_factory=list)
    max_calories: float | None = None
    max_time: float | None = None
    category: str | None = None
    top_n: int = Field(default=12, ge=1, le=50)


def clean_value(value):
    """Convertit les valeurs pandas/numpy en valeurs JSON."""
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): clean_value(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]

    if hasattr(value, "item"):
        value = value.item()

    if pd.isna(value):
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def find_recipe_details(recipe_id):
    """Récupère les détails complémentaires depuis le catalogue du modèle."""
    recipes = model.recipes

    matches = recipes[
        recipes["RecipeId"].astype(str) == str(recipe_id)
    ]

    if matches.empty:
        return {}

    recipe = matches.iloc[0]

    fields = [
        "Description",
        "RecipeInstructions",
        "RecipeServings",
        "RecipeIngredientQuantities",
    ]

    return {
        field: clean_value(recipe[field])
        for field in fields
        if field in recipes.columns
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/api/recommendations")
def get_recommendations(request: RecommendationRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé."
        )

    if not request.wanted_ingredients and not request.category:
        raise HTTPException(
            status_code=400,
            detail="Saisissez au moins un ingrédient ou choisissez une catégorie."
        )

    try:
        results = model.recommend(
            wanted_ingredients=request.wanted_ingredients,
            avoided_ingredients=request.avoided_ingredients,
            max_calories=request.max_calories,
            max_time=request.max_time,
            category=request.category,
            top_n=request.top_n
        )

        recipes = []

        for _, row in results.iterrows():
            recipe = {
                key: clean_value(value)
                for key, value in row.to_dict().items()
            }

            recipe.update(
                find_recipe_details(recipe.get("RecipeId"))
            )

            recipes.append(recipe)

        return {
            "count": len(recipes),
            "recipes": recipes
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Erreur recommandation :", error)

        raise HTTPException(
            status_code=500,
            detail="Une erreur est survenue pendant la recommandation."
        )
# ============================================================
# AJOUT : Taste Discovery (preference_model.pkl)
#
# À coller À LA FIN de ton fichier d'API qui fonctionne.
# Rien du code existant n'est modifié ni supprimé.
# ============================================================

import ast
import re

# Si cet import échoue, l'API démarre quand même
# (seule la route taste-discovery sera indisponible).
try:
    from src.recommendation.preference_clustering import recommend_for_user
except ImportError as error:
    recommend_for_user = None
    print(f"Taste Discovery indisponible : {error}")


PREFERENCE_MODEL_PATH = ROOT_DIR / "models" / "preference_model.pkl"

preference_model = None
preference_recipe_to_idx = {}
catalog_position = {}


@app.on_event("startup")
def load_preference_model():

    global preference_model, preference_recipe_to_idx, catalog_position

    try:
        if not PREFERENCE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {PREFERENCE_MODEL_PATH}"
            )

        print("Chargement du modèle Preference Clustering...")

        with open(PREFERENCE_MODEL_PATH, "rb") as file:
            preference_model = pickle.load(file)

        # RecipeId -> index dans le modèle de préférences.
        preference_recipe_to_idx = {
            str(recipe_id): idx
            for idx, recipe_id in enumerate(preference_model["recipe_ids"])
        }

        # RecipeId -> ligne dans le catalogue du modèle content-based
        # (sert à récupérer le nom, les ingrédients, les étapes, etc.).
        catalog_position = {
            str(recipe_id): position
            for position, recipe_id in enumerate(model.recipes["RecipeId"])
        }

        print("Modèle Preference Clustering chargé.")

    except Exception as error:
        preference_model = None
        print(f"Modèle de préférences non chargé : {error}")


def parse_text_list(value):
    """Transforme une liste, une liste en texte ou un vecteur R en liste."""

    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass

    # Format R : c("élément 1", "élément 2")
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', text)

    return quoted if quoted else [text]


def build_taste_recipe(recipe_id):
    """Construit la fiche recette à partir du catalogue content-based."""

    position = catalog_position.get(str(recipe_id))

    if position is None:
        return None

    recipes = model.recipes
    row = recipes.iloc[position]

    if "IngredientsList" in recipes.columns:
        ingredients = row["IngredientsList"]
    else:
        ingredients = row.get("RecipeIngredientParts")

    return {
        "RecipeId": str(recipe_id),
        "Name": clean_value(row.get("Name")),
        "RecipeCategory": clean_value(row.get("RecipeCategory")),
        "RecipeIngredientParts": parse_text_list(ingredients),
        "CaloriesValue": clean_value(row.get("CaloriesValue")),
        "TimeMinutes": clean_value(row.get("TimeMinutes")),
        "Description": clean_value(row.get("Description")),
        "RecipeInstructions": parse_text_list(
            row.get("RecipeInstructions")
        ),
        "RecipeServings": clean_value(row.get("RecipeServings")),
    }


class TasteRequest(BaseModel):

    liked_ids: list[str] = Field(default_factory=list)
    disliked_ids: list[str] = Field(default_factory=list)
    seen_ids: list[str] = Field(default_factory=list)


@app.post("/api/taste-discovery/next")
def next_taste_recipe(request: TasteRequest):

    if preference_model is None or recommend_for_user is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle de préférences n'est pas chargé."
        )

    def to_indices(recipe_ids):
        return [
            preference_recipe_to_idx[recipe_id]
            for recipe_id in recipe_ids
            if recipe_id in preference_recipe_to_idx
        ]

    try:
        recommended_indices, cluster_id = recommend_for_user(
            liked_indices=to_indices(request.liked_ids),
            disliked_indices=to_indices(request.disliked_ids),
            model_data=preference_model,
            top_n=20,
            seen_indices=to_indices(request.seen_ids)
        )

        # On prend la première recette dont on a les informations.
        for idx in recommended_indices:

            recipe_id = str(preference_model["recipe_ids"][idx])
            recipe = build_taste_recipe(recipe_id)

            if recipe is not None:
                return {
                    "finished": False,
                    "recipe": recipe,
                    "cluster_id": cluster_id
                }

        return {
            "finished": True,
            "recipe": None,
            "cluster_id": cluster_id
        }

    except Exception as error:
        print("Erreur Taste Discovery :", error)

        raise HTTPException(
            status_code=500,
            detail="Impossible de générer une recommandation."
        )

# ============================================================
# 6. RECOMMANDATIONS PERSONNALISEES FINALES
# ============================================================

class PersonalizedTasteRequest(BaseModel):
    liked_ids: list[str] = Field(default_factory=list)
    disliked_ids: list[str] = Field(default_factory=list)
    seen_ids: list[str] = Field(default_factory=list)


@app.post("/api/taste-discovery/personalized")
def get_personalized_taste_recommendations(
    request: PersonalizedTasteRequest
):
    if preference_model is None or recommend_for_user is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle de préférences n'est pas chargé."
        )

    try:
        # Conversion des RecipeId en indices du modèle.
        def to_indices(recipe_ids):
            return list(dict.fromkeys(
                preference_recipe_to_idx[ str(recipe_id) ]
                for recipe_id in recipe_ids
                if str(recipe_id) in preference_recipe_to_idx
            ))

        liked_indices = to_indices(request.liked_ids)
        disliked_indices = to_indices(request.disliked_ids)
        seen_indices = to_indices(request.seen_ids)

        # On demande jusqu'à 50 candidats pour pouvoir récupérer
        # 10 fiches complètes même si certains candidats manquent
        # dans le catalogue utilisé par le frontend.
        recommended_indices, cluster_id = recommend_for_user(
            liked_indices=liked_indices,
            disliked_indices=disliked_indices,
            model_data=preference_model,
            top_n=50,
            seen_indices=seen_indices
        )

        recipes = []

        for idx in recommended_indices:
            recipe_id = str(preference_model["recipe_ids"][idx])

            recipe = build_taste_recipe(recipe_id)

            if recipe is not None:
                recipe["RecommendationRank"] = len(recipes) + 1
                recipes.append(recipe)

            if len(recipes) == 10:
                break

        return {
            "count": len(recipes),
            "recipes": recipes,
            "cluster_id": cluster_id,
            "message": (
                "Recommendations generated successfully."
                if recipes
                else "No additional recommendations are available."
            )
        }

    except Exception as error:
        print("Erreur recommandations personnalisées :", error)

        raise HTTPException(
            status_code=500,
            detail="Impossible de générer les recommandations personnalisées."
        )
    