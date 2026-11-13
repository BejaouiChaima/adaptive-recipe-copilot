
import { useEffect, useState } from "react";
import {
  checkApi,
  getRecommendations,
  logRecommendations,
  logReaction,
  type Preferences,
  type Recipe,
} from "./api";
import TasteDiscovery from "./TasteDiscovery";

type Page = "home" | "discover" | "preferences" | "chef" | "taste";
type Reaction = "like" | "dislike";

const emptyPreferences: Preferences = {
  wanted_ingredients: [],
  avoided_ingredients: [],
  category: "",
  max_calories: null,
  max_time: null,
};

const categories = [
  "Dessert",
  "Lunch",
  "Dinner",
  "Breakfast",
  "Side Dishes",
  "Appetizers",
  "Salad",
  "Beverages",
];

function splitIngredients(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatIngredients(value?: string | string[] | null): string {
  if (!value) return "Ingrédients non disponibles";

  if (Array.isArray(value)) {
    return value.join(", ");
  }

  // Nettoyage des formats de type c("ingredient", "ingredient")
  return value
    .replace(/^c\(/, "")
    .replace(/\)$/, "")
    .replace(/[\[\]']/g, "")
    .replace(/"/g, "")
    .trim();
}

function formatNumber(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "Non renseigné";
  }

  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

// --------------------------------------------------
// CARTE RECETTE AVEC LIKE / DISLIKE
// --------------------------------------------------

function RecipeCard({ recipe }: { recipe: Recipe }) {
  const ingredients = formatIngredients(
    recipe.RecipeIngredientParts
  );

  const [reaction, setReaction] = useState<Reaction | null>(null);
  const [sendingReaction, setSendingReaction] = useState(false);

  async function handleReaction(value: Reaction) {
    if (sendingReaction) return;

    setSendingReaction(true);
    setReaction(value);

    try {
      await logReaction("content_based", recipe, value);
    } finally {
      setSendingReaction(false);
    }
  }

  return (
    <article className="recipe-card">
      <div className="recipe-card-top">
        <span className="recipe-category">
          {recipe.RecipeCategory || "Recipe"}
        </span>

        <span className="recipe-symbol">✳</span>
      </div>

      <h3>{recipe.Name}</h3>

      {recipe.Description && (
        <p className="recipe-description">
          {recipe.Description.length > 125
            ? `${recipe.Description.slice(0, 125)}...`
            : recipe.Description}
        </p>
      )}

      <div className="recipe-details">
        <div>
          <span className="detail-icon">◷</span>
          <span>{formatNumber(recipe.TimeMinutes)} min</span>
        </div>

        <div>
          <span className="detail-icon">♡</span>
          <span>{formatNumber(recipe.CaloriesValue)} kcal</span>
        </div>
      </div>

      <div className="ingredients">
        <span className="ingredients-label">INGREDIENTS</span>

        <p>
          {ingredients.length > 150
            ? `${ingredients.slice(0, 150)}...`
            : ingredients}
        </p>
      </div>

      {/* BOUTONS DE REACTION */}

      <div className="recipe-reactions">
        <button
          type="button"
          className={
            reaction === "like"
              ? "reaction-button selected"
              : "reaction-button"
          }
          onClick={() => handleReaction("like")}
          disabled={sendingReaction}
          aria-pressed={reaction === "like"}
        >
          ♥ Like
        </button>

        <button
          type="button"
          className={
            reaction === "dislike"
              ? "reaction-button selected"
              : "reaction-button"
          }
          onClick={() => handleReaction("dislike")}
          disabled={sendingReaction}
          aria-pressed={reaction === "dislike"}
        >
          ✕ Dislike
        </button>
      </div>

      {reaction && (
        <p className="reaction-feedback">
          {reaction === "like"
            ? "You liked this recipe."
            : "You passed on this recipe."}
        </p>
      )}
    </article>
  );
}

// --------------------------------------------------
// APPLICATION PRINCIPALE
// --------------------------------------------------

function App() {
  const [page, setPage] = useState<Page>("home");
  const [hasCompleted, setHasCompleted] = useState(false);

  const [preferences, setPreferences] =
    useState<Preferences>(emptyPreferences);

  const [wantedText, setWantedText] = useState("");
  const [avoidedText, setAvoidedText] = useState("");

  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);

  // --------------------------------------------------
  // CHARGEMENT DES PREFERENCES SAUVEGARDEES
  // --------------------------------------------------

  useEffect(() => {
    const saved = localStorage.getItem("recipe-preferences");

    if (saved) {
      try {
        const parsed = JSON.parse(saved) as Preferences;

        setPreferences(parsed);

        setWantedText(
          parsed.wanted_ingredients.join(", ")
        );

        setAvoidedText(
          parsed.avoided_ingredients.join(", ")
        );

        setHasCompleted(true);

        // Recharger les recommandations après actualisation
        loadRecipes(parsed);
      } catch {
        localStorage.removeItem("recipe-preferences");
      }
    }
  }, []);

  // --------------------------------------------------
  // VERIFICATION DE L'API
  // --------------------------------------------------

  useEffect(() => {
    checkApi().then(setApiOnline);
  }, []);

  // --------------------------------------------------
  // CHARGER LES RECOMMANDATIONS CONTENT-BASED
  // --------------------------------------------------

  async function loadRecipes(
    currentPreferences: Preferences
  ) {
    setLoading(true);
    setError("");

    try {
      const results = await getRecommendations(
        currentPreferences
      );

      setRecipes(results);

      // ENREGISTRER LES RECETTES AFFICHEES
      await logRecommendations(
        "content_based",
        results
      );
    } catch (err) {
      setRecipes([]);

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
  // SAUVEGARDER LES PREFERENCES
  // --------------------------------------------------

  function savePreferences() {
    const updated: Preferences = {
      wanted_ingredients: splitIngredients(wantedText),
      avoided_ingredients: splitIngredients(avoidedText),
      category: preferences.category,
      max_calories: preferences.max_calories,
      max_time: preferences.max_time,
    };

    if (
      !updated.wanted_ingredients.length &&
      !updated.category
    ) {
      setError(
        "Ajoute au moins un ingrédient ou sélectionne une catégorie."
      );
      return;
    }

    setPreferences(updated);

    localStorage.setItem(
      "recipe-preferences",
      JSON.stringify(updated)
    );

    setHasCompleted(true);
    setError("");
    setPage("home");

    loadRecipes(updated);
  }

  // --------------------------------------------------
  // MODIFIER LES LIMITES NUMERIQUES
  // --------------------------------------------------

  function updateNumber(
    field: "max_calories" | "max_time",
    value: string
  ) {
    setPreferences((current) => ({
      ...current,
      [field]: value === "" ? null : Number(value),
    }));
  }

  // --------------------------------------------------
  // OUVRIR LES PREFERENCES
  // --------------------------------------------------

  function openPreferences() {
    setWantedText(
      preferences.wanted_ingredients.join(", ")
    );

    setAvoidedText(
      preferences.avoided_ingredients.join(", ")
    );

    setPage("preferences");
  }

  // --------------------------------------------------
  // DEMARRER L'ONBOARDING
  // --------------------------------------------------

  function startOnboarding() {
    setWantedText("");
    setAvoidedText("");
    setPreferences(emptyPreferences);
    setError("");
    setPage("preferences");
  }

  // --------------------------------------------------
  // AFFICHER LES RECETTES
  // --------------------------------------------------

  function renderRecipes() {
    if (loading) {
      return (
        <div className="message-state">
          <div className="loader" />
          <p>Nous préparons vos recommandations...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="message-state error-state">
          <span>!</span>
          <p>{error}</p>

          <button
            className="secondary-button"
            onClick={() => loadRecipes(preferences)}
          >
            Réessayer
          </button>
        </div>
      );
    }

    if (!recipes.length) {
      return (
        <div className="message-state">
          <span className="empty-symbol">✳</span>

          <h3>Aucune recette pour le moment</h3>

          <p>
            Modifie tes ingrédients ou tes préférences
            pour essayer à nouveau.
          </p>

          <button
            className="secondary-button"
            onClick={openPreferences}
          >
            Modifier mes préférences
          </button>
        </div>
      );
    }

    return (
      <div className="recipe-grid">
        {recipes.map((recipe) => (
          <RecipeCard
            key={recipe.RecipeId}
            recipe={recipe}
          />
        ))}
      </div>
    );
  }

  // --------------------------------------------------
  // PAGE D'ACCUEIL AVANT CONFIGURATION
  // --------------------------------------------------

  if (
    !hasCompleted &&
    page !== "preferences" &&
    page !== "taste"
  ) {
    return (
      <main className="welcome-screen">
        <div className="welcome-content">
          <div className="brand-mark">✳</div>

          <p className="eyebrow">
            YOUR PERSONAL RECIPE COMPANION
          </p>

          <h1>
            Let's make something <em>delicious.</em>
          </h1>

          <p className="welcome-description">
            Discover recipes that match your ingredients,
            your preferences and your everyday cooking habits.
          </p>

          <button
            className="primary-button"
            onClick={startOnboarding}
          >
            Get started <span>→</span>
          </button>

          <p className="welcome-note">
            A little inspiration for every meal.
          </p>
        </div>

        <div className="welcome-decoration">
          <span className="decoration-circle circle-one" />
          <span className="decoration-circle circle-two" />
          <span className="decoration-circle circle-three" />

          <div className="decoration-text">
            GOOD FOOD
            <br />
            GOOD MOOD
          </div>
        </div>
      </main>
    );
  }

  // --------------------------------------------------
  // APPLICATION
  // --------------------------------------------------

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">✳</div>

          <div>
            <strong>crumb.</strong>
            <span>RECIPE COMPANION</span>
          </div>
        </div>

        <p className="nav-label">MENU</p>

        <nav className="navigation">
          <button
            className={
              page === "home"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setPage("home")}
          >
            <span>⌂</span> Home
          </button>

          <button
            className={
              page === "discover"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setPage("discover")}
          >
            <span>⌕</span> Discover recipes
          </button>

          <button
            className={
              page === "taste"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setPage("taste")}
          >
            <span>✦</span> Taste Discovery
          </button>

          <button
            className={
              page === "preferences"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={openPreferences}
          >
            <span>♡</span> My preferences
          </button>

          <button
            className={
              page === "chef"
                ? "nav-item active"
                : "nav-item"
            }
            onClick={() => setPage("chef")}
          >
            <span>✧</span> AI Chef
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-tip">
            <span>✳</span>
            <p>
              Good things take time. Enjoy the process.
            </p>
          </div>

          <div className="api-status">
            <span
              className={
                apiOnline
                  ? "status-dot online"
                  : "status-dot"
              }
            />

            {apiOnline
              ? "Recipe service connected"
              : "Recipe service unavailable"}
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            YOUR KITCHEN / {page.toUpperCase()}
          </div>

          <button
            className="profile-button"
            onClick={openPreferences}
          >
            <span className="profile-avatar">C</span>
            <span>My profile</span>
          </button>
        </header>

        {/* HOME */}

        {page === "home" && (
          <section className="page-section">
            <div className="hero-banner">
              <div>
                <p className="eyebrow">
                  A FRESH START IN YOUR KITCHEN
                </p>

                <h1>
                  Good food starts
                  <br />
                  with <em>what you love.</em>
                </h1>

                <p>
                  Discover something delicious to make today.
                </p>

                <button
                  className="dark-button"
                  onClick={() => setPage("discover")}
                >
                  Explore recipes <span>→</span>
                </button>
              </div>

              <div className="hero-art">
                <div className="hero-art-inner">✳</div>

                <span>
                  MADE FOR
                  <br />
                  YOUR TASTE
                </span>
              </div>
            </div>

            <div className="section-heading">
              <div>
                <p className="eyebrow">PICKED FOR YOU</p>

                <h2>Your recipe inspiration</h2>

                <p className="section-subtitle">
                  Suggestions based on your preferences.
                </p>
              </div>

              <button
                className="text-button"
                onClick={() => setPage("discover")}
              >
                View all <span>→</span>
              </button>
            </div>

            {renderRecipes()}
          </section>
        )}

        {/* DISCOVER */}

        {page === "discover" && (
          <section className="page-section">
            <p className="eyebrow">
              FIND YOUR NEXT FAVORITE
            </p>

            <h1 className="page-title">
              Discover recipes<span>.</span>
            </h1>

            <p className="page-intro">
              Explore dishes that fit what you have
              and what you enjoy.
            </p>

            <div className="filter-panel">
              <div className="filter-summary">
                <span>YOUR CURRENT SEARCH</span>

                <p>
                  {preferences.wanted_ingredients.length
                    ? preferences.wanted_ingredients.join(", ")
                    : "Your selected category"}
                </p>
              </div>

              <button
                className="secondary-button"
                onClick={openPreferences}
              >
                Edit filters
              </button>
            </div>

            {renderRecipes()}
          </section>
        )}

        {/* TASTE DISCOVERY */}

        {page === "taste" && (
          <section className="page-section taste-page-section">
            <TasteDiscovery />
          </section>
        )}

        {/* PREFERENCES */}

        {page === "preferences" && (
          <section className="page-section form-section">
            <p className="eyebrow">
              LET'S GET TO KNOW YOUR TASTE
            </p>

            <h1 className="page-title">
              Your kitchen,
              <br />
              <em>your preferences.</em>
            </h1>

            <p className="page-intro">
              Tell us what you like. You can always change
              these choices later.
            </p>

            <div className="preferences-form">
              <div className="form-group">
                <label htmlFor="wanted">
                  Which ingredients do you enjoy?
                </label>

                <p>Separate ingredients with commas.</p>

                <input
                  id="wanted"
                  value={wantedText}
                  onChange={(event) =>
                    setWantedText(event.target.value)
                  }
                  placeholder="e.g. chicken, tomato, cheese"
                />
              </div>

              <div className="form-group">
                <label htmlFor="avoided">
                  Any ingredients you'd rather avoid?
                </label>

                <p>
                  We'll exclude recipes containing these ingredients.
                </p>

                <input
                  id="avoided"
                  value={avoidedText}
                  onChange={(event) =>
                    setAvoidedText(event.target.value)
                  }
                  placeholder="e.g. mushrooms, peanuts"
                />
              </div>

              <div className="form-group">
                <label htmlFor="category">
                  What kind of recipes interest you?
                </label>

                <select
                  id="category"
                  value={preferences.category}
                  onChange={(event) =>
                    setPreferences((current) => ({
                      ...current,
                      category: event.target.value,
                    }))
                  }
                >
                  <option value="">
                    Choose a category
                  </option>

                  {categories.map((category) => (
                    <option
                      key={category}
                      value={category}
                    >
                      {category}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="time">
                    Maximum cooking time (minutes)
                  </label>

                  <input
                    id="time"
                    type="number"
                    min="1"
                    value={preferences.max_time ?? ""}
                    onChange={(event) =>
                      updateNumber(
                        "max_time",
                        event.target.value
                      )
                    }
                    placeholder="e.g. 45"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="calories">
                    Maximum calories
                  </label>

                  <input
                    id="calories"
                    type="number"
                    min="1"
                    value={preferences.max_calories ?? ""}
                    onChange={(event) =>
                      updateNumber(
                        "max_calories",
                        event.target.value
                      )
                    }
                    placeholder="e.g. 500"
                  />
                </div>
              </div>

              {error && (
                <p className="form-error">{error}</p>
              )}

              <button
                className="primary-button save-button"
                onClick={savePreferences}
              >
                {hasCompleted
                  ? "Save my preferences"
                  : "Discover my recipes"}

                <span>→</span>
              </button>
            </div>
          </section>
        )}

        {/* AI CHEF */}

        {page === "chef" && (
          <section className="page-section chef-section">
            <p className="eyebrow">
              A LITTLE HELP IN THE KITCHEN
            </p>

            <h1 className="page-title">
              Meet your AI Chef<span>.</span>
            </h1>

            <div className="chef-card">
              <div className="chef-symbol">✧</div>

              <h2>
                Your cooking companion is coming soon.
              </h2>

              <p>
                Soon, you will be able to ask cooking questions,
                understand recipe instructions and get help
                with ingredients.
              </p>

              <span className="coming-soon">
                COMING SOON
              </span>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
