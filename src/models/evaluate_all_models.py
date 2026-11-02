
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "recommendation"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train_reviews.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test_reviews.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 10
MAX_USERS = 500
RANDOM_STATE = 42


# ==================================================
# 2. LOAD MODELS AND DATA
# ==================================================

print("Loading models...")

content_model = joblib.load(MODEL_DIR / "content_model.pkl")
svd_model = joblib.load(MODEL_DIR / "svd_model.pkl")
hybrid_model = joblib.load(MODEL_DIR / "hybrid_model.pkl")

print("Loading datasets...")

train_reviews = pd.read_csv(TRAIN_PATH)
test_reviews = pd.read_csv(TEST_PATH)

# Normalize IDs to strings
for df in [train_reviews, test_reviews]:
    df["AuthorId"] = df["AuthorId"].astype(str)
    df["RecipeId"] = df["RecipeId"].astype(str)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

train_reviews = train_reviews.dropna(
    subset=["AuthorId", "RecipeId", "Rating"]
)

test_reviews = test_reviews.dropna(
    subset=["AuthorId", "RecipeId", "Rating"]
)

print("Train reviews:", len(train_reviews))
print("Test reviews:", len(test_reviews))


# ==================================================
# 3. PREPARE KNOWN LIKES AND RELEVANT RECIPES
# ==================================================

# Known likes: positive interactions in Train
train_positive = train_reviews[
    train_reviews["Rating"] >= 4
].copy()

# Relevant recipes: positive interactions in Test
test_positive = test_reviews[
    test_reviews["Rating"] >= 4
].copy()

# Users with exactly 5 known positive recipes
likes_per_user = train_positive.groupby("AuthorId").size()

eligible_users = set(
    likes_per_user[likes_per_user == 5].index
)

# SVD can only recommend recipes that appeared in its training data.
# Use a common evaluation catalog for a fair comparison.
svd_recipe_ids = set(train_reviews["RecipeId"].unique())

test_positive = test_positive[
    test_positive["RecipeId"].isin(svd_recipe_ids)
].copy()

# Prepare known likes by user
known_likes = {}

for user_id, group in train_positive.groupby("AuthorId"):
    if user_id in eligible_users:
        known_likes[user_id] = group

# Prepare relevant test recipes by user
relevant_by_user = (
    test_positive.groupby("AuthorId")["RecipeId"]
    .apply(set)
    .to_dict()
)

# Keep users who have both 5 known likes and relevant test recipes
evaluation_users = [
    user_id
    for user_id in eligible_users
    if user_id in relevant_by_user
    and len(relevant_by_user[user_id]) > 0
]

print("\nEligible users with relevant test recipes:", len(evaluation_users))

if len(evaluation_users) == 0:
    raise ValueError("No users available for evaluation.")

# Deterministic sample
rng = np.random.RandomState(RANDOM_STATE)

if len(evaluation_users) > MAX_USERS:
    evaluation_users = rng.choice(
        evaluation_users,
        size=MAX_USERS,
        replace=False
    ).tolist()

print("Users selected for evaluation:", len(evaluation_users))
print("Top K:", TOP_K)


# ==================================================
# 4. METRIC FUNCTIONS
# ==================================================

def calculate_metrics(recommended_ids, relevant_ids, k=10):

    recommended_ids = list(dict.fromkeys(recommended_ids))[:k]

    relevant_ids = set(relevant_ids)

    hits = sum(
        1 for recipe_id in recommended_ids
        if recipe_id in relevant_ids
    )

    # Precision@K
    precision = hits / k

    # Recall@K
    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    # F1@K
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # NDCG@K (binary relevance)
    dcg = 0.0

    for rank, recipe_id in enumerate(recommended_ids):
        if recipe_id in relevant_ids:
            dcg += 1.0 / np.log2(rank + 2)

    ideal_hits = min(k, len(relevant_ids))

    idcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, f1, ndcg


# ==================================================
# 5. EVALUATE ALL THREE MODELS
# ==================================================

model_names = [
    "Content-Based",
    "SVD",
    "Hybrid"
]

user_metrics = {
    model_name: []
    for model_name in model_names
}

print("\nStarting evaluation...")

for user_id in tqdm(evaluation_users):

    # The user's 5 known liked recipes
    user_likes = known_likes[user_id]

    liked_recipe_ids = user_likes["RecipeId"].tolist()
    liked_ratings = user_likes["Rating"].tolist()

    # Relevant recipes from Test
    relevant_ids = relevant_by_user[user_id]

    # Exclude known likes from relevant recipes
    relevant_ids = relevant_ids - set(liked_recipe_ids)

    if not relevant_ids:
        continue

    # Content-Based
    content_results = content_model.recommend(
        liked_recipe_ids,
        top_n=TOP_K
    )

    # SVD
    svd_results = svd_model.recommend(
        liked_recipe_ids,
        liked_ratings=liked_ratings,
        top_n=TOP_K
    )

    # Hybrid
    hybrid_results = hybrid_model.recommend(
        liked_recipe_ids,
        liked_ratings=liked_ratings,
        top_n=TOP_K
    )

    results_by_model = {
        "Content-Based": content_results,
        "SVD": svd_results,
        "Hybrid": hybrid_results
    }

    # Calculate metrics
    for model_name, results in results_by_model.items():

        if results is None or results.empty:
            recommended_ids = []
        else:
            recommended_ids = (
                results["RecipeId"].astype(str).tolist()
            )

        metrics = calculate_metrics(
            recommended_ids,
            relevant_ids,
            TOP_K
        )

        user_metrics[model_name].append(metrics)


# ==================================================
# 6. CALCULATE AVERAGE RESULTS
# ==================================================

summary = []

for model_name in model_names:

    metrics_list = user_metrics[model_name]

    if not metrics_list:
        continue

    metrics_array = np.array(metrics_list)

    summary.append({
        "Model": model_name,
        "Precision@10": metrics_array[:, 0].mean(),
        "Recall@10": metrics_array[:, 1].mean(),
        "F1@10": metrics_array[:, 2].mean(),
        "NDCG@10": metrics_array[:, 3].mean(),
        "Evaluated Users": len(metrics_list)
    })

summary_df = pd.DataFrame(summary)

# Display percentages for readability
print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "Precision@10": "{:.4f}".format,
            "Recall@10": "{:.4f}".format,
            "F1@10": "{:.4f}".format,
            "NDCG@10": "{:.4f}".format
        }
    )
)

# Save results
output_path = RESULTS_DIR / "model_comparison.csv"

summary_df.to_csv(output_path, index=False)

print("\nEvaluation completed.")
print("Results saved to:", output_path)
