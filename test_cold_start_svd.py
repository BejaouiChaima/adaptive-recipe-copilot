
from src.recommendation.content_based_svd import ContentBasedSVDRecommender
import pandas as pd

# 1. Charger les données et les modèles une seule fois
model = ContentBasedSVDRecommender(
    max_recipes=60000,
    max_users=2500,
    n_components=50,
    content_weight=0.5,
    random_state=42,
).load()

# 2. Tester plusieurs quantités d'informations sur l'utilisateur
scenarios = [
    {"n_likes": 3, "n_test": 3},
    {"n_likes": 5, "n_test": 5},
    {"n_likes": 8, "n_test": 5},
]

all_summaries = []
all_details = []

for scenario in scenarios:
    n_likes = scenario["n_likes"]
    n_test = scenario["n_test"]

    print(
        f"\n=== TEST : {n_likes} recettes connues, "
        f"{n_test} recettes cachées ==="
    )

    summary, details = model.evaluate_cold_start(
        n_users=50,
        n_likes=n_likes,
        n_test=n_test,
        top_k=10,
        random_state=42,
    )

    # Ajouter les paramètres du scénario aux résultats
    summary = summary.copy()
    summary["N_Likes"] = n_likes
    summary["N_Test"] = n_test
    all_summaries.append(summary)

    details = details.copy()
    details["N_Likes"] = n_likes
    details["N_Test"] = n_test
    all_details.append(details)

    print(summary.round(4).to_string(index=False))

# 3. Regrouper les résultats
summary_final = pd.concat(all_summaries, ignore_index=True)
details_final = pd.concat(all_details, ignore_index=True)

# 4. Enregistrer les résultats
summary_final.to_csv(
    "cold_start_svd_summary.csv",
    index=False,
    encoding="utf-8",
)

details_final.to_csv(
    "cold_start_svd_details.csv",
    index=False,
    encoding="utf-8",
)

# 5. Afficher la comparaison finale
print("\n=== COMPARAISON FINALE ===")

columns_summary = [
    "N_Likes",
    "N_Test",
    "Model",
    "UsersEvaluated",
    "UsersWithHit",
    "TotalHits",
    "Precision@K",
    "Recall@K",
    "F1@K",
    "NDCG@K",
]

# Afficher uniquement les colonnes réellement présentes
columns_summary = [
    c for c in columns_summary if c in summary_final.columns
]

print(
    summary_final[columns_summary]
    .round(4)
    .to_string(index=False)
)

print("\nFichiers créés :")
print("- cold_start_svd_summary.csv")
print("- cold_start_svd_details.csv")
