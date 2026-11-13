
const API_URL = "http://127.0.0.1:8000";

export interface Preferences {
  wanted_ingredients: string[];
  avoided_ingredients: string[];
  category: string;
  max_calories: number | null;
  max_time: number | null;
}

export interface Recipe {
  RecipeId: number | string;
  Name: string;
  RecipeCategory?: string | null;
  RecipeIngredientParts?: string | string[] | null;
  CaloriesValue?: number | null;
  TimeMinutes?: number | null;
  Description?: string | null;
  RecipeInstructions?: string | string[] | null;
  RecipeServings?: number | null;
  FinalScore?: number | null;
}

interface RecommendationResponse {
  count: number;
  recipes: Recipe[];
}

export type EvaluationAlgorithm =
  | "content_based"
  | "taste_discovery";

export type UserReaction = "like" | "dislike";

export function getSessionId(): string {
  const key = "recipe-evaluation-session";
  let sessionId = localStorage.getItem(key);

  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(key, sessionId);
  }

  return sessionId;
}

export async function checkApi(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/api/health`);
    if (!response.ok) return false;

    const data = await response.json();
    return data.status === "ok" && data.model_loaded === true;
  } catch {
    return false;
  }
}

export async function getRecommendations(
  preferences: Preferences
): Promise<Recipe[]> {
  const response = await fetch(`${API_URL}/api/recommendations`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...preferences,
      category: preferences.category || null,
      top_n: 12,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Impossible de récupérer les recettes.");
  }

  return (data as RecommendationResponse).recipes;
}

export async function logRecommendations(
  algorithm: EvaluationAlgorithm,
  recipes: Recipe[]
): Promise<void> {
  if (!recipes.length) return;

  try {
    const response = await fetch(`${API_URL}/api/evaluation/exposures`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: getSessionId(),
        algorithm,
        recipes: recipes.map((recipe) => ({
          recipe_id: String(recipe.RecipeId),
          recipe_name: recipe.Name || "",
        })),
      }),
    });

    if (!response.ok) {
      console.error("Erreur lors de l'enregistrement des recommandations.");
    }
  } catch (error) {
    console.error("Suivi des recommandations indisponible :", error);
  }
}

export async function logReaction(
  algorithm: EvaluationAlgorithm,
  recipe: Recipe,
  reaction: UserReaction
): Promise<void> {
  try {
    const response = await fetch(`${API_URL}/api/evaluation/reactions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: getSessionId(),
        algorithm,
        recipe_id: String(recipe.RecipeId),
        recipe_name: recipe.Name || "",
        reaction,
      }),
    });

    if (!response.ok) {
      console.error("Erreur lors de l'enregistrement de la réaction.");
    }
  } catch (error) {
    console.error("Suivi des réactions indisponible :", error);
  }
}
