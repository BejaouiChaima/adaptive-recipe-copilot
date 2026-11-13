
import { useEffect, useState } from "react";
import "./TasteDiscovery.css";

import {
  logRecommendations,
  logReaction,
  type Recipe,
} from "./api";

const API_URL = "http://127.0.0.1:8000";
const MAX_RECIPES = 12;

// Type utilisé par Taste Discovery.
// Il reprend Recipe depuis api.ts et ajoute RecommendationRank.
type TasteRecipe = Recipe & {
  RecommendationRank?: number;
};

interface RecommendationResponse {
  finished: boolean;
  recipe: TasteRecipe | null;
  cluster_id: number | null;
}

interface PersonalizedResponse {
  count: number;
  recipes: TasteRecipe[];
  cluster_id: number | null;
  message: string;
}

type Reaction = "like" | "dislike";

function formatList(value?: string[] | string | null): string[] {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  return [value];
}

function RecipeDetails({ recipe }: { recipe: TasteRecipe }) {
  const ingredients = formatList(recipe.RecipeIngredientParts);
  const instructions = formatList(recipe.RecipeInstructions);

  return (
    <div className="taste-recipe-content">
      <h2>{recipe.Name}</h2>

      {recipe.Description && (
        <p className="taste-description">{recipe.Description}</p>
      )}

      <div className="taste-recipe-stats">
        {recipe.CaloriesValue != null && (
          <div className="taste-stat">
            <span className="taste-stat-icon">◉</span>
            <div>
              <strong>{Math.round(recipe.CaloriesValue)}</strong>
              <small>Calories</small>
            </div>
          </div>
        )}

        {recipe.TimeMinutes != null && (
          <div className="taste-stat">
            <span className="taste-stat-icon">◷</span>
            <div>
              <strong>{recipe.TimeMinutes} min</strong>
              <small>Preparation time</small>
            </div>
          </div>
        )}

        {recipe.RecipeServings != null && (
          <div className="taste-stat">
            <span className="taste-stat-icon">♧</span>
            <div>
              <strong>{recipe.RecipeServings}</strong>
              <small>Servings</small>
            </div>
          </div>
        )}
      </div>

      {ingredients.length > 0 && (
        <div className="taste-detail-section">
          <h3>Ingredients</h3>

          <div className="taste-ingredients">
            {ingredients.map((ingredient, index) => (
              <span key={`${ingredient}-${index}`}>
                {ingredient}
              </span>
            ))}
          </div>
        </div>
      )}

      {instructions.length > 0 && (
        <details className="taste-instructions">
          <summary>View preparation instructions</summary>

          <ol>
            {instructions.map((instruction, index) => (
              <li key={index}>{instruction}</li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}

export default function TasteDiscovery() {
  const [recipe, setRecipe] = useState<TasteRecipe | null>(null);

  const [likedIds, setLikedIds] = useState<string[]>([]);
  const [dislikedIds, setDislikedIds] = useState<string[]>([]);
  const [seenIds, setSeenIds] = useState<string[]>([]);

  const [loading, setLoading] = useState(true);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState("");

  const [lastReaction, setLastReaction] = useState<Reaction | null>(null);

  // Recommandations personnalisées
  const [personalizedRecipes, setPersonalizedRecipes] =
    useState<TasteRecipe[]>([]);

  const [personalizedLoading, setPersonalizedLoading] = useState(false);
  const [personalizedError, setPersonalizedError] = useState("");
  const [showPersonalized, setShowPersonalized] = useState(false);
  const [clusterId, setClusterId] = useState<number | null>(null);

  // --------------------------------------------------
  // CHARGER LA PROCHAINE RECETTE
  // --------------------------------------------------

  async function fetchNextRecipe(
    likes: string[],
    dislikes: string[],
    seen: string[]
  ) {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_URL}/api/taste-discovery/next`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            liked_ids: likes,
            disliked_ids: dislikes,
            seen_ids: seen,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Impossible de charger une recette."
        );
      }

      const result = data as RecommendationResponse;

      if (result.finished || !result.recipe) {
        setRecipe(null);
        setFinished(true);
        return;
      }

      setRecipe(result.recipe);
      setFinished(false);

      // ENREGISTRER LA RECETTE AFFICHÉE
      await logRecommendations(
        "taste_discovery",
        [result.recipe]
      );

    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Une erreur est survenue."
      );
    } finally {
      setLoading(false);
    }
  }

  // --------------------------------------------------
  // GENERER LES RECOMMANDATIONS PERSONNALISEES
  // --------------------------------------------------

  async function fetchPersonalizedRecommendations() {
    setPersonalizedLoading(true);
    setPersonalizedError("");
    setShowPersonalized(false);

    try {
      const response = await fetch(
        `${API_URL}/api/taste-discovery/personalized`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            liked_ids: likedIds,
            disliked_ids: dislikedIds,
            seen_ids: seenIds,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Impossible de générer vos recommandations."
        );
      }

      const result = data as PersonalizedResponse;

      const recommendations = result.recipes || [];

      setPersonalizedRecipes(recommendations);
      setClusterId(result.cluster_id);
      setShowPersonalized(true);

      // ENREGISTRER LES RECOMMANDATIONS FINALES
      await logRecommendations(
        "taste_discovery",
        recommendations
      );

    } catch (err) {
      setPersonalizedError(
        err instanceof Error
          ? err.message
          : "Une erreur est survenue pendant la génération."
      );
    } finally {
      setPersonalizedLoading(false);
    }
  }

  // --------------------------------------------------
  // PREMIERE RECETTE AU CHARGEMENT
  // --------------------------------------------------

  useEffect(() => {
    fetchNextRecipe([], [], []);
  }, []);

  // --------------------------------------------------
  // GERER LIKE / DISLIKE
  // --------------------------------------------------

  async function handleReaction(reaction: Reaction) {
    if (!recipe || loading) return;

    const recipeId = String(recipe.RecipeId);

    const nextSeen = [...seenIds, recipeId];

    const nextLikes =
      reaction === "like"
        ? [...likedIds, recipeId]
        : likedIds;

    const nextDislikes =
      reaction === "dislike"
        ? [...dislikedIds, recipeId]
        : dislikedIds;

    // ENREGISTRER LA REACTION
    await logReaction(
      "taste_discovery",
      recipe,
      reaction
    );

    setSeenIds(nextSeen);
    setLikedIds(nextLikes);
    setDislikedIds(nextDislikes);
    setLastReaction(reaction);

    // FIN DE LA SESSION APRES 5 RECETTES
    if (nextSeen.length >= MAX_RECIPES) {
      setRecipe(null);
      setFinished(true);
      return;
    }

    // CHARGER LA RECETTE SUIVANTE
    await fetchNextRecipe(
      nextLikes,
      nextDislikes,
      nextSeen
    );
  }

  // --------------------------------------------------
  // RECOMMENCER UNE SESSION
  // --------------------------------------------------

  function restartDiscovery() {
    setLikedIds([]);
    setDislikedIds([]);
    setSeenIds([]);

    setLastReaction(null);

    setRecipe(null);
    setFinished(false);
    setError("");

    setPersonalizedRecipes([]);
    setPersonalizedError("");
    setShowPersonalized(false);
    setClusterId(null);

    fetchNextRecipe([], [], []);
  }

  // --------------------------------------------------
  // AFFICHAGE
  // --------------------------------------------------

  return (
    <main className="taste-page">

      <header className="taste-header">
        <div>
          <span className="taste-eyebrow">
            PERSONALIZED DISCOVERY
          </span>

          <h1>Discover your taste</h1>

          <p>
            Explore recipes one by one. Your reactions help us understand
            what you enjoy and improve your next suggestions.
          </p>
        </div>

        <div className="taste-counter">
          <span className="taste-counter-number">
            {seenIds.length}
          </span>

          <span>recipes explored</span>
        </div>
      </header>

      {/* PROGRESSION */}

      <section className="taste-progress-section">
        <div className="taste-progress-labels">
          <span>Your discovery</span>

          <span>
            {likedIds.length} liked · {dislikedIds.length} passed
          </span>
        </div>

        <div className="taste-progress-track">
          <div
            className="taste-progress-fill"
            style={{
              width: `${Math.min(
                (seenIds.length / MAX_RECIPES) * 100,
                100
              )}%`,
            }}
          />
        </div>
      </section>

      {/* CHARGEMENT */}

      {loading && (
        <section className="taste-state">
          <div className="taste-spinner" />

          <h2>Finding your next recipe...</h2>

          <p>
            Updating your recommendations based on your reactions.
          </p>
        </section>
      )}

      {/* ERREUR */}

      {!loading && error && (
        <section className="taste-state taste-error">
          <h2>Something went wrong</h2>

          <p>{error}</p>

          <button
            className="taste-secondary-button"
            onClick={() =>
              fetchNextRecipe(
                likedIds,
                dislikedIds,
                seenIds
              )
            }
          >
            Try again
          </button>
        </section>
      )}

      {/* FIN DE LA SESSION */}

      {!loading &&
        !error &&
        finished &&
        !showPersonalized &&
        !personalizedLoading && (
          <section className="taste-state taste-finished">

            <div className="taste-finished-icon">✓</div>

            <h2>You've completed your discovery session!</h2>

            <p>
              You explored {seenIds.length} recipes, liked{" "}
              {likedIds.length} and passed on {dislikedIds.length}.
            </p>

            <p>
              Your reactions are ready. Let us find recipes
              that match your taste.
            </p>

            {personalizedError && (
              <p className="taste-personalized-error">
                {personalizedError}
              </p>
            )}

            <button
              className="taste-primary-button"
              onClick={fetchPersonalizedRecommendations}
              disabled={personalizedLoading}
            >
              {personalizedLoading
                ? "Finding your recipes..."
                : "Get My Personalized Recommendations"}
            </button>

            <button
              className="taste-secondary-button"
              onClick={restartDiscovery}
              disabled={personalizedLoading}
            >
              Start a new discovery session
            </button>

          </section>
        )}

      {/* GENERATION DES RECOMMANDATIONS */}

      {personalizedLoading && (
        <section className="taste-state">
          <div className="taste-spinner" />

          <h2>Creating your personalized list...</h2>

          <p>
            Our recommendation system is analyzing your likes
            and dislikes.
          </p>
        </section>
      )}

      {/* RESULTATS PERSONNALISES */}

      {showPersonalized && !personalizedLoading && (
        <section className="taste-personalized-section">

          <div className="taste-personalized-header">
            <span className="taste-eyebrow">
              YOUR PERSONALIZED RESULTS
            </span>

            <h2>Recipes selected for your taste</h2>

            <p>
              Based on your {likedIds.length} likes and{" "}
              {dislikedIds.length} passed recipes.
            </p>

            <span className="taste-result-count">
              {personalizedRecipes.length} recommendations
            </span>
          </div>

          {personalizedRecipes.length === 0 ? (
            <div className="taste-state">

              <div className="taste-finished-icon">✓</div>

              <h2>No additional recipes found</h2>

              <p>
                There are no more unseen recipes available for your
                current preferences. Try a new discovery session.
              </p>

            </div>
          ) : (
            <div className="taste-personalized-grid">

              {personalizedRecipes.map((item, index) => (
                <article
                  className="taste-personalized-card"
                  key={String(item.RecipeId)}
                >
                  <div className="taste-card-top">

                    <span className="taste-recipe-label">
                      RECOMMENDATION #
                      {item.RecommendationRank || index + 1}
                    </span>

                    {item.RecipeCategory && (
                      <span className="taste-category">
                        {item.RecipeCategory}
                      </span>
                    )}

                  </div>

                  <RecipeDetails recipe={item} />

                </article>
              ))}

            </div>
          )}

          {clusterId !== null && (
            <p className="taste-cluster-note">
              Your preference profile was matched with taste group #
              {clusterId}.
            </p>
          )}

          <div className="taste-personalized-actions">
            <button
              className="taste-primary-button"
              onClick={restartDiscovery}
            >
              Start a new discovery session
            </button>
          </div>

        </section>
      )}

      {/* RECETTE EN COURS */}

      {!loading && !error && recipe && !finished && (
        <section className="taste-card">

          <div className="taste-card-top">

            <span className="taste-recipe-label">
              RECIPE SUGGESTION
            </span>

            {recipe.RecipeCategory && (
              <span className="taste-category">
                {recipe.RecipeCategory}
              </span>
            )}

          </div>

          <RecipeDetails recipe={recipe} />

          <div className="taste-reaction-area">

            <p>
              Would you like to see more recipes like this?
            </p>

            <div className="taste-reaction-buttons">

              <button
                className="taste-dislike-button"
                onClick={() => handleReaction("dislike")}
                disabled={loading}
              >
                <span>✕</span>
                Not for me
              </button>

              <button
                className="taste-like-button"
                onClick={() => handleReaction("like")}
                disabled={loading}
              >
                <span>♥</span>
                I like this
              </button>

            </div>

            {lastReaction && (
              <small className="taste-last-reaction">
                Your previous reaction:{" "}
                {lastReaction === "like" ? "Liked" : "Passed"}
              </small>
            )}

          </div>
        </section>
      )}

      {/* FOOTER */}

      <footer className="taste-footer">
        <span>{likedIds.length} likes</span>

        <span className="taste-footer-dot">•</span>

        <span>{dislikedIds.length} passed</span>

        <span className="taste-footer-dot">•</span>

        <span>{seenIds.length} explored</span>
      </footer>

    </main>
  );
}
