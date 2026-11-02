
from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "recommendation"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train_reviews.csv"
RECIPES_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_clean.csv"


# 1. Load models
print("Loading models...")

content_model = joblib.load(MODEL_DIR / "content_model.pkl")
svd_model = joblib.load(MODEL_DIR / "svd_model.pkl")
hybrid_model = joblib.load(MODEL_DIR / "hybrid_model.pkl")


# 2. Load training data
train_reviews = pd.read_csv(TRAIN_PATH)
recipes = pd.read_csv(RECIPES_PATH)

# Positive interactions: rating >= 4
positive_reviews = train_reviews[
    train_reviews["Rating"] >= 4
].copy()

# Select a user with exactly 5 positive interactions
user_counts = positive_reviews.groupby("AuthorId").size()

eligible_users = user_counts[user_counts == 5].index

if len(eligible_users) == 0:
    raise ValueError("No user with exactly 5 positive interactions found.")

user_id = eligible_users[0]

user_likes = positive_reviews[
    positive_reviews["AuthorId"] == user_id
]

liked_recipe_ids = user_likes["RecipeId"].astype(str).tolist()
liked_ratings = user_likes["Rating"].tolist()

print("\nSelected user:", user_id)
print("Number of liked recipes:", len(liked_recipe_ids))


# 3. Content-Based recommendations
content_results = content_model.recommend(
    liked_recipe_ids,
    top_n=10
)


# 4. SVD recommendations
svd_results = svd_model.recommend(
    liked_recipe_ids,
    liked_ratings=liked_ratings,
    top_n=10
)


# 5. Hybrid recommendations
hybrid_results = hybrid_model.recommend(
    liked_recipe_ids,
    liked_ratings=liked_ratings,
    top_n=10
)


# 6. Display recommendations with recipe names
def display_results(title, results):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if results.empty:
        print("No recommendations found.")
        return

    result_ids = results["RecipeId"].astype(str)

    names = recipes.copy()
    names["RecipeId"] = names["RecipeId"].astype(str)

    output = results.merge(
        names[["RecipeId", "Name", "RecipeCategory"]],
        on="RecipeId",
        how="left"
    )

    print(output.to_string(index=False))


display_results("CONTENT-BASED RECOMMENDATIONS", content_results)
display_results("SVD RECOMMENDATIONS", svd_results)
display_results("HYBRID RECOMMENDATIONS", hybrid_results)
