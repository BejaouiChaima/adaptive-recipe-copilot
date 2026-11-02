
from pathlib import Path
import joblib
import pandas as pd

from recommendation_models import (
    ContentBasedModel,
    SVDRecommender,
    HybridRecommender
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECIPES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "recipes_clean.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "splits" / "train_reviews.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "recommendation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading datasets...")

recipes = pd.read_csv(RECIPES_PATH)
train_reviews = pd.read_csv(TRAIN_PATH)

print("Recipes:", len(recipes))
print("Train reviews:", len(train_reviews))


# =====================================================
# MODEL 1: CONTENT-BASED
# =====================================================

content_model = ContentBasedModel(
    max_features=100000
)

content_model.fit(recipes)

joblib.dump(
    content_model,
    MODEL_DIR / "content_model.pkl"
)


# =====================================================
# MODEL 2: SVD
# =====================================================

svd_model = SVDRecommender(
    n_components=50
)

svd_model.fit(train_reviews)

joblib.dump(
    svd_model,
    MODEL_DIR / "svd_model.pkl"
)


# =====================================================
# MODEL 3: HYBRID
# =====================================================

hybrid_model = HybridRecommender(
    content_model=content_model,
    svd_model=svd_model,
    content_weight=0.5,
    candidate_pool=500
)

joblib.dump(
    hybrid_model,
    MODEL_DIR / "hybrid_model.pkl"
)


print("\nAll three models have been trained and saved.")
print("Saved in:", MODEL_DIR)
