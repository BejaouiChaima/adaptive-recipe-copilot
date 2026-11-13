
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "evaluation_logs"
OUTPUT_DIR = BASE_DIR / "tests" / "evaluation_results"

RECOMMENDATIONS_FILE = LOG_DIR / "recommendations.csv"
REACTIONS_FILE = LOG_DIR / "reactions.csv"

ALGORITHMS = {
    "content_based": "Content-Based",
    "taste_discovery": "Taste Discovery",
}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recommendations = read_csv(RECOMMENDATIONS_FILE)
    reactions = read_csv(REACTIONS_FILE)

    if recommendations.empty:
        print("Aucune recommandation enregistrée.")
        print("Utilise d'abord l'application pour collecter des données.")
        return

    if reactions.empty:
        print("Aucune réaction enregistrée.")
        print("Les recommandations sont enregistrées, mais aucun Like/Dislike.")
        return

    required_recommendation_columns = {
        "session_id",
        "algorithm",
        "recipe_id",
    }

    required_reaction_columns = {
        "session_id",
        "algorithm",
        "recipe_id",
        "reaction",
    }

    if not required_recommendation_columns.issubset(
        recommendations.columns
    ):
        raise ValueError("Le fichier recommendations.csv est incomplet.")

    if not required_reaction_columns.issubset(reactions.columns):
        raise ValueError("Le fichier reactions.csv est incomplet.")

    recommendations["recipe_id"] = (
        recommendations["recipe_id"].astype(str)
    )

    reactions["recipe_id"] = reactions["recipe_id"].astype(str)

    recommendations["algorithm"] = (
        recommendations["algorithm"].astype(str)
    )

    reactions["algorithm"] = reactions["algorithm"].astype(str)

    reactions["reaction"] = (
        reactions["reaction"].astype(str).str.lower()
    )

    # Une seule réaction finale par recette et par session.
    reactions = reactions.drop_duplicates(
        subset=["session_id", "algorithm", "recipe_id"],
        keep="last",
    )

    results = []

    for algorithm, label in ALGORITHMS.items():

        displayed = recommendations[
            recommendations["algorithm"] == algorithm
        ]

        feedback = reactions[
            reactions["algorithm"] == algorithm
        ]

        likes = int((feedback["reaction"] == "like").sum())
        dislikes = int((feedback["reaction"] == "dislike").sum())

        total_feedback = likes + dislikes
        total_displayed = len(displayed)

        acceptance_rate = (
            likes / total_feedback * 100
            if total_feedback
            else 0
        )

        reaction_rate = (
            total_feedback / total_displayed * 100
            if total_displayed
            else 0
        )

        like_rate = (
            likes / total_displayed * 100
            if total_displayed
            else 0
        )

        unique_recipes = displayed["recipe_id"].nunique()
        sessions = displayed["session_id"].nunique()

        results.append({
            "Algorithm": label,
            "Total recommendations displayed": total_displayed,
            "Unique recipes displayed": unique_recipes,
            "Sessions": sessions,
            "Likes": likes,
            "Dislikes": dislikes,
            "Total feedback": total_feedback,
            "Acceptance rate among feedback (%)": round(
                acceptance_rate, 2
            ),
            "Feedback rate (%)": round(
                reaction_rate, 2
            ),
            "Likes / displayed recommendations (%)": round(
                like_rate, 2
            ),
        })

    report = pd.DataFrame(results)

    report_path = OUTPUT_DIR / "online_evaluation_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("       ONLINE RECOMMENDATION EVALUATION")
    print("=" * 65)

    print(report.to_string(index=False))

    print("\nRapport sauvegardé dans :")
    print(report_path)

    # Export des données nettoyées.
    recommendations.to_csv(
        OUTPUT_DIR / "recommendations_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reactions.to_csv(
        OUTPUT_DIR / "reactions_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
