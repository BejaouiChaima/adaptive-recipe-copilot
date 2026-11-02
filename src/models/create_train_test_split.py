
from pathlib import Path
import pandas as pd

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REVIEWS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "reviews_clean.csv"
)

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train_reviews.csv"
TEST_PATH = SPLIT_DIR / "test_reviews.csv"

# =====================================================
# CONFIGURATION
# =====================================================

RANDOM_STATE = 42
N_LIKES = 5
MIN_POSITIVE_RATINGS = 6

# =====================================================
# LOAD REVIEWS
# =====================================================

print("Loading reviews...")

reviews = pd.read_csv(REVIEWS_PATH)

required_columns = ["AuthorId", "RecipeId", "Rating"]

for column in required_columns:
    if column not in reviews.columns:
        raise ValueError(f"Missing column: {column}")

reviews["AuthorId"] = reviews["AuthorId"].astype(str)
reviews["RecipeId"] = reviews["RecipeId"].astype(str)
reviews["Rating"] = pd.to_numeric(
    reviews["Rating"], errors="coerce"
)

reviews = reviews.dropna(
    subset=["AuthorId", "RecipeId", "Rating"]
)

# Aggregate duplicate user-recipe interactions
reviews = reviews.groupby(
    ["AuthorId", "RecipeId"],
    as_index=False
)["Rating"].mean()

print("Total reviews:", len(reviews))
print("Total users:", reviews["AuthorId"].nunique())

# =====================================================
# POSITIVE AND NON-POSITIVE INTERACTIONS
# =====================================================

positive = reviews[reviews["Rating"] >= 4].copy()
non_positive = reviews[reviews["Rating"] < 4].copy()

positive_counts = positive.groupby(
    "AuthorId"
)["RecipeId"].nunique()

eligible_users = positive_counts[
    positive_counts >= MIN_POSITIVE_RATINGS
].index

print("Eligible users:", len(eligible_users))

# =====================================================
# USER-BASED TRAIN / TEST SPLIT
# =====================================================

known_parts = []
test_parts = []

for user_id, user_data in positive[
    positive["AuthorId"].isin(eligible_users)
].groupby("AuthorId"):

    # Shuffle each user's positive interactions
    user_data = user_data.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    # Five known liked recipes
    known = user_data.iloc[:N_LIKES]

    # Remaining positive recipes are hidden
    hidden = user_data.iloc[N_LIKES:]

    known_parts.append(known)
    test_parts.append(hidden)

known_reviews = pd.concat(
    known_parts,
    ignore_index=True
)

test_reviews = pd.concat(
    test_parts,
    ignore_index=True
)

# Keep all interactions that are not hidden in Test

ineligible_positive = positive[
    ~positive["AuthorId"].isin(eligible_users)
].copy()

train_reviews = pd.concat(
    [
        known_reviews,
        non_positive,
        ineligible_positive
    ],
    ignore_index=True
)

train_reviews = train_reviews.drop_duplicates(
    subset=["AuthorId", "RecipeId"]
)

# =====================================================
# SAVE FILES
# =====================================================

train_reviews.to_csv(TRAIN_PATH, index=False)
test_reviews.to_csv(TEST_PATH, index=False)

print("\nTrain/Test split completed.")
print("Train reviews:", len(train_reviews))
print("Test reviews:", len(test_reviews))
print("Users evaluated:", len(eligible_users))

print("\nFiles created:")
print(TRAIN_PATH)
print(TEST_PATH)
