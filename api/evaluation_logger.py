
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "evaluation_logs"

RECOMMENDATIONS_FILE = LOG_DIR / "recommendations.csv"
REACTIONS_FILE = LOG_DIR / "reactions.csv"

CSV_LOCK = threading.Lock()

RECOMMENDATION_FIELDS = [
    "timestamp",
    "session_id",
    "algorithm",
    "recipe_id",
    "recipe_name",
]

REACTION_FIELDS = [
    "timestamp",
    "session_id",
    "algorithm",
    "recipe_id",
    "recipe_name",
    "reaction",
]


class RecommendationItem(BaseModel):
    recipe_id: str
    recipe_name: str = ""


class ExposureRequest(BaseModel):
    session_id: str
    algorithm: Literal["content_based", "taste_discovery"]
    recipes: list[RecommendationItem] = Field(default_factory=list)


class ReactionRequest(BaseModel):
    session_id: str
    algorithm: Literal["content_based", "taste_discovery"]
    recipe_id: str
    recipe_name: str = ""
    reaction: Literal["like", "dislike"]


def ensure_csv(path: Path, fields: list[str]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()


def append_rows(path: Path, fields: list[str], rows: list[dict]):
    if not rows:
        return

    with CSV_LOCK:
        ensure_csv(path, fields)

        with path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writerows(rows)


@router.post("/exposures")
def log_exposures(payload: ExposureRequest):
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = [
        {
            "timestamp": timestamp,
            "session_id": payload.session_id,
            "algorithm": payload.algorithm,
            "recipe_id": recipe.recipe_id,
            "recipe_name": recipe.recipe_name,
        }
        for recipe in payload.recipes
    ]

    append_rows(
        RECOMMENDATIONS_FILE,
        RECOMMENDATION_FIELDS,
        rows,
    )

    return {
        "status": "ok",
        "logged": len(rows),
    }


@router.post("/reactions")
def log_reaction(payload: ReactionRequest):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": payload.session_id,
        "algorithm": payload.algorithm,
        "recipe_id": payload.recipe_id,
        "recipe_name": payload.recipe_name,
        "reaction": payload.reaction,
    }

    append_rows(
        REACTIONS_FILE,
        REACTION_FIELDS,
        [row],
    )

    return {"status": "ok"}
