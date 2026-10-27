
from src.recommendation.hybride import ReviewTransferRecommender
# Charger les données et construire le modèle
model = ReviewTransferRecommender(
    max_recipes=60000,
    max_users=2500
).load()

# Simuler le cold start sur 50 utilisateurs
summary, details = model.evaluate_cold_start(
    n_users=50,
    n_likes=3,
    n_test=3,
    top_k=10,
    random_state=42
)

# Sauvegarder les résultats détaillés
details.to_csv("cold_start_results.csv", index=False)

print("\nRésumé des métriques :")
print(summary)

print("\nExemples de résultats par utilisateur :")
print(details.head())
