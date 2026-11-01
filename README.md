# adaptive-recipe-copilot

## Classification de sentiment des avis (LLM)

Les avis de `data/processed/reviews_clean.csv` sont classés en positif (1) /
négatif (0) par un LLM via l'API OpenRouter, avec reprise possible après une
coupure (quota, crash, Ctrl+C).

### 1. Configuration

Copier `.env.example` en `.env` puis renseigner la clé :

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
```

`.env` est ignoré par git : la clé ne doit jamais être committée.

### 2. Préparer le fichier d'entrée

```
python src/data/prepare_reviews_for_llm.py
```

Génère `data/processed/reviews_for_llm.csv` avec 3 colonnes seulement
(`ReviewId`, `Review`, `Rating`). Aucune ligne n'est supprimée : la ligne N du
fichier correspond à la ligne N de `reviews_clean.csv`.

### 3. Lancer la classification

```
# Aperçu du prompt + estimation du coût, sans appel API
python src/data/classify_reviews_llm.py --limit 100 --dry-run

# Traiter 100 avis (simulation "jour 1")
python src/data/classify_reviews_llm.py --limit 100

# Reprise automatique à la ligne 101 (simulation "jour 2")
python src/data/classify_reviews_llm.py --limit 100

# Forcer un point de départ
python src/data/classify_reviews_llm.py --start-line 5000 --limit 1000

# Tout recommencer
python src/data/classify_reviews_llm.py --overwrite --limit 100
```

Sorties :

| Fichier | Contenu |
|---|---|
| `data/processed/reviews_sentiment.csv` | `ReviewId, Review, Rating, Sentiment` |
| `data/processed/llm_sentiment_checkpoint.json` | dernière ligne terminée, coût cumulé |

Valeurs de `Sentiment` :

- `1` : avis positif
- `0` : avis négatif
- `-1` : exception (quota / tokens épuisés, erreur API, réponse illisible)

### Options utiles

| Option | Défaut | Rôle |
|---|---|---|
| `--model` | `google/gemini-2.5-flash-lite` | modèle OpenRouter |
| `--fallback-models` | `openai/gpt-4.1-nano,mistralai/mistral-small-3.2-24b-instruct` | modèles de secours si le principal est saturé (`""` pour désactiver) |
| `--batch-size` | `10` | avis par requête (réduit fortement le coût) |
| `--workers` | `3` | requêtes en parallèle (baisser en cas de 429) |
| `--save-every` | `100` | écriture dans le CSV tous les N avis |
| `--max-chars` | `1200` | troncature du texte des avis |
| `--max-retries` | `6` | tentatives par requête (backoff exponentiel) |
| `--start-line` | auto | 1ère ligne à traiter (1 = début) |
| `--limit` | toutes | nombre de lignes de ce run |

### Erreur 429 (rate limit)

`google/gemini-2.5-flash-lite is temporarily rate-limited upstream` : le quota
du fournisseur est atteint, ce n'est pas une erreur du script.

1. Le script bascule automatiquement sur un modèle de secours
   (`--fallback-models`).
2. Il respecte l'en-tête `Retry-After` et attend jusqu'à 60 s entre deux essais.
3. Si les 429 persistent : `--workers 1` et/ou `--batch-size 5`.
4. Rien n'est perdu : chaque tranche de `--save-every` avis est déjà écrite.

### Coût indicatif

Avec `google/gemini-2.5-flash-lite` (0,10 $/M tokens entrée) et
`--batch-size 10`, le coût mesuré est d'environ **11 à 15 $ pour les 1,4 M
d'avis**. Le coût réel de chaque run est affiché à la fin et cumulé dans le
checkpoint.

### Comportement en cas d'erreur

- Erreur transitoire (429, 5xx, timeout) : retries avec backoff exponentiel,
  bascule sur un modèle de secours, respect de `Retry-After`.
- Lot illisible : repli avis par avis, puis `-1` si l'échec persiste.
- Clé invalide / crédit épuisé (401, 402, 403) : arrêt immédiat, aucun avis
  n'est écrit, le checkpoint reste sur la dernière ligne terminée.
- 3 lots consécutifs en échec : arrêt du run (quota épuisé ou service HS).
- Ctrl+C : les lots déjà terminés sont écrits avant l'arrêt.

## Analyse de sentiment — approche NLP (sans API)

Le notebook `NLP/01_sentiment_nlp.ipynb` reproduit la même classification avec
des méthodes NLP classiques, gratuites et locales :

1. **VADER** — lexique de sentiment (règles, négations, intensificateurs).
2. **TF-IDF + régression logistique** — modèle supervisé entraîné sur les notes.

| | Entrée | Sortie |
|---|---|---|
| Fichier | `data/processed/reviews_for_llm.csv` | `data/processed/reviews_nlp_sentiment.csv` |
| Colonnes | `ReviewId, Review, Rating` | `ReviewId, Review, Rating, Sentiment` |

Le format de `Sentiment` est identique au pipeline LLM (`1` / `0` / `-1`), les
deux fichiers se joignent donc sur `ReviewId`.

```powershell
pip install vaderSentiment
python -m jupyter notebook NLP/01_sentiment_nlp.ipynb
```

Le traitement est incrémental (checkpoint `nlp_sentiment_checkpoint.json`) :
un notebook interrompu reprend à la ligne suivante. Pour tester sur un
échantillon :

```powershell
$env:NB_ROW_LIMIT="20000"
python -m jupyter notebook NLP/01_sentiment_nlp.ipynb
```

Résultats mesurés sur 3 000 avis :

| Méthode | Métrique |
|---|---|
| VADER vs note | 87,4 % d'accord |
| TF-IDF + LogReg (200 k avis) | 96,2 % d'accuracy |
| VADER vs LLM | 90,7 % d'accord |
