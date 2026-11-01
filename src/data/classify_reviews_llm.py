"""
Classification de sentiment des avis par LLM (OpenRouter).

Entrée  : data/processed/reviews_for_llm.csv   (ReviewId, Review, Rating)
Sortie  : data/processed/reviews_sentiment.csv (ReviewId, Review, Rating, Sentiment)

Sentiment :
    1  -> avis positif
    0  -> avis négatif
   -1  -> exception (quota / tokens épuisés, erreur API, réponse illisible)

Reprise après coupure :
    Chaque bloc traité est écrit immédiatement dans le CSV de sortie et la
    progression est sauvegardée dans data/processed/llm_sentiment_checkpoint.json.
    Si le run s'arrête (quota, Ctrl+C, crash), le run suivant reprend
    automatiquement à la ligne suivante. On peut aussi forcer le point de
    départ :

        --start-line 101     # reprendre à la ligne 101 (1 = première ligne)
        --limit 100          # ne traiter que 100 lignes (simule un quota/jour)
        --overwrite          # tout recommencer depuis zéro

Exemples :
    # Simulation "jour 1" : 100 avis
    python src/data/classify_reviews_llm.py --limit 100

    # Simulation "jour 2" : reprend automatiquement à la ligne 101
    python src/data/classify_reviews_llm.py --limit 100

    # Vérifier le prompt et estimer le coût sans appeler l'API
    python src/data/classify_reviews_llm.py --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "reviews_for_llm.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reviews_sentiment.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "processed" / "llm_sentiment_checkpoint.json"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
# Utilisés automatiquement par OpenRouter si le modèle principal est saturé.
DEFAULT_FALLBACK_MODELS = "openai/gpt-4.1-nano,mistralai/mistral-small-3.2-24b-instruct"

# Prix indicatifs en $ / million de tokens (uniquement pour l'estimation --dry-run).
MODEL_PRICES = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-5-nano": (0.05, 0.40),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "mistralai/mistral-small-3.2-24b-instruct": (0.075, 0.20),
    "qwen/qwen3-30b-a3b-instruct-2507": (0.10, 0.30),
    "deepseek/deepseek-chat-v3.1": (0.25, 0.95),
}

SYSTEM_PROMPT = (
    "You are a sentiment classifier for recipe reviews. "
    "You always answer with strict JSON and nothing else."
)

USER_PROMPT_TEMPLATE = """Classify the sentiment of each recipe review below.

Labels:
- 1 = positive (satisfied, thanks, recommends, it worked)
- 0 = negative (disappointed, complaint, it failed, too sweet/bland/dry)

Rules:
- Base the decision on the review text.
- If the text is empty or too short to judge, use the rating: 4-5 => 1, 0-2 => 0, 3 => 1.
- For mixed reviews, follow the dominant tone.
- Ignore the review number, the rating and the quote characters.

Return ONLY this JSON, with one integer per review, in the same order:
{{"s": [1, 0, ...]}}

Reviews:
{reviews}"""

COLUMNS = ["ReviewId", "Review", "Rating", "Sentiment"]


class FatalAPIError(RuntimeError):
    """Erreur non récupérable (clé invalide, crédit épuisé, service HS...)."""


class APIError(requests.HTTPError):
    """Erreur HTTP récupérable (429, 5xx...), avec Retry-After éventuel."""

    def __init__(self, message: str, response=None, retry_after: float | None = None):
        super().__init__(message, response=response)
        self.retry_after = retry_after


class SentimentClassifier:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0,
                 max_retries: int = 6, max_chars: int = 1200,
                 timeout: int = 60, base_sleep: float = 2.0,
                 max_consecutive_failures: int = 3, verbose: bool = True,
                 fallback_models: list[str] | None = None):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.timeout = timeout
        self.base_sleep = base_sleep
        self.max_consecutive_failures = max_consecutive_failures
        self.verbose = verbose
        self.fallback_models = fallback_models or []

        self._local = threading.local()
        self._lock = threading.Lock()
        self.total_cost = 0.0
        self.consecutive_failures = 0
        self.served_models: dict[str, int] = {}

    # ------------------------------------------------------------------ HTTP
    @property
    def _session(self) -> requests.Session:
        # Une session par thread (le pool de connexions n'est pas thread-safe).
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _request(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/BejaouiChaima/adaptive-recipe-copilot",
            "X-Title": "adaptive-recipe-copilot",
        }
        response = self._session.post(API_URL, headers=headers, json=payload,
                                      timeout=self.timeout)
        if response.status_code in (401, 402, 403):
            raise FatalAPIError(
                f"HTTP {response.status_code} : {response.text[:300]}\n"
                "Vérifiez la clé OPENROUTER_API_KEY et le crédit du compte."
            )
        if response.status_code != 200:
            retry_after = response.headers.get("Retry-After")
            raise APIError(
                f"HTTP {response.status_code} : {response.text[:200]}",
                response=response,
                retry_after=float(retry_after) if retry_after else None,
            )
        return response.json()

    # --------------------------------------------------------------- prompts
    def build_prompt(self, records: list[dict]) -> str:
        lines = []
        for i, rec in enumerate(records, start=1):
            text = (rec["Review"] or "").strip().replace('"', "'")
            if len(text) > self.max_chars:
                text = text[: self.max_chars] + "..."
            rating = "?" if rec["Rating"] is None else int(rec["Rating"])
            lines.append(f'[{i}] (rating {rating}) "{text}"')
        return USER_PROMPT_TEMPLATE.format(reviews="\n".join(lines))

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _parse_labels(content: str, expected: int) -> list[int] | None:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[-1] if "\n" in text else text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        labels = data.get("s") if isinstance(data, dict) else data
        if not isinstance(labels, list) or len(labels) != expected:
            return None
        try:
            labels = [int(x) for x in labels]
        except (TypeError, ValueError):
            return None
        if any(x not in (0, 1) for x in labels):
            return None
        return labels

    def _call(self, records: list[dict]) -> list[int] | None:
        """Un appel API, avec retries. None si échec définitif."""
        payload = {
            "temperature": self.temperature,
            "max_tokens": 8 * len(records) + 20,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.build_prompt(records)},
            ],
        }
        # OpenRouter bascule automatiquement sur le modèle suivant si le
        # premier est indisponible ou rate-limited en amont.
        if self.fallback_models:
            payload["models"] = [self.model, *self.fallback_models]
        else:
            payload["model"] = self.model

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                data = self._request(payload)
                content = data["choices"][0]["message"].get("content", "")
                with self._lock:
                    self.total_cost += float(data.get("usage", {}).get("cost") or 0.0)
                    served = data.get("model") or self.model
                    self.served_models[served] = self.served_models.get(served, 0) + 1

                labels = self._parse_labels(content, len(records))
                if labels is not None:
                    return labels
                last_error = f"réponse illisible : {content[:200]!r}"
                delay = self.base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            except FatalAPIError:
                raise
            except APIError as exc:
                last_error = f"HTTP {exc.response.status_code}: {str(exc)[:160]}"
                # 429 / 5xx : attente longue, en respectant Retry-After.
                delay = exc.retry_after or min(
                    self.base_sleep * (2 ** attempt) + random.uniform(0, 1.0), 60.0)
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                delay = self.base_sleep * (2 ** attempt) + random.uniform(0, 0.5)

            if attempt < self.max_retries:
                if self.verbose:
                    print(f"    retry {attempt + 1}/{self.max_retries} "
                          f"dans {delay:.1f}s ({last_error})", flush=True)
                time.sleep(delay)

        if self.verbose:
            print(f"    ECHEC definitif : {last_error}", flush=True)
        return None

    # ---------------------------------------------------------- classification
    def classify_batch(self, records: list[dict]) -> list[int]:
        """Retourne une liste de labels (0/1/-1) de la taille du lot."""
        labels = self._call(records)

        # Repli : si le lot échoue et qu'il contient plusieurs avis, on réessaie
        # avis par avis pour ne pas perdre tout le lot.
        if labels is None and len(records) > 1:
            labels = []
            for rec in records:
                single = self._call([rec])
                labels.append(single[0] if single else -1)

        if labels is None:
            labels = [-1] * len(records)

        with self._lock:
            if -1 in labels:
                self.consecutive_failures += 1
            else:
                self.consecutive_failures = 0
            failures = self.consecutive_failures

        if failures >= self.max_consecutive_failures:
            raise FatalAPIError(
                f"{failures} lots consécutifs en échec - arrêt du run "
                "(quota épuisé ou service indisponible)."
            )
        return labels


# --------------------------------------------------------------------- I/O
def iter_parts(path: Path, start_line: int, limit: int | None,
               read_chunk: int = 20000):
    """Itère sur des blocs de lignes à partir de `start_line` (1-based)."""
    reader = pd.read_csv(
        path,
        chunksize=read_chunk,
        dtype={"ReviewId": "Int64", "Review": "string", "Rating": "Int64"},
        keep_default_na=False,
        na_values=[""],
    )
    remaining = limit
    offset = 0  # nombre de lignes déjà lues depuis le début du fichier
    for chunk in reader:
        chunk = chunk.reset_index(drop=True)
        begin = max(0, (start_line - 1) - offset)
        end = len(chunk)
        if remaining is not None:
            end = min(end, begin + remaining)
        if begin < end:
            part = chunk.iloc[begin:end].copy()
            part_start = offset + begin + 1
            if remaining is not None:
                remaining -= len(part)
            yield part, part_start
        offset += len(chunk)
        if remaining is not None and remaining <= 0:
            break


def count_data_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "ReviewId": None if pd.isna(r.ReviewId) else int(r.ReviewId),
            "Review": "" if pd.isna(r.Review) else str(r.Review),
            "Rating": None if pd.isna(r.Rating) else int(r.Rating),
        }
        for r in df.itertuples(index=False)
    ]


def estimate_cost_per_row(records: list[dict], model: str,
                          batch_size: int) -> float | None:
    """Coût moyen estimé par avis, en dollars."""
    prices = MODEL_PRICES.get(model)
    if not prices or not records:
        return None
    avg_chars = sum(len(r["Review"]) for r in records) / len(records)
    n_batches = max(len(records) / max(batch_size, 1), 1)
    input_tokens = (avg_chars / 4 + 15) * len(records) + 120 * n_batches
    output_tokens = 3 * len(records) + 10 * n_batches
    cost = input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1]
    return cost / len(records)


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    # Évite un crash si la console Windows n'encode pas certains caractères.
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Classifie le sentiment des avis avec un LLM via OpenRouter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--start-line", type=int, default=None,
                        help="1ère ligne à traiter (1 = début du fichier). "
                             "Par défaut : reprise depuis le checkpoint.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximum de lignes à traiter dans ce run.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Nombre d'avis envoyés dans un même prompt.")
    parser.add_argument("--workers", type=int, default=3,
                        help="Requêtes API en parallèle (baisser si erreurs 429).")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Écriture dans le CSV tous les N avis (sécurité).")
    parser.add_argument("--fallback-models", type=str, default=DEFAULT_FALLBACK_MODELS,
                        help="Modèles de secours (séparés par des virgules) utilisés "
                             "si le modèle principal est rate-limited. Vide = aucun.")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=1200,
                        help="Troncature du texte de l'avis.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Repartir de zéro (vide le CSV de sortie).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche le prompt et le coût estimé, sans appel API.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    args.fallback_models = [m.strip() for m in (args.fallback_models or "").split(",")
                            if m.strip() and m.strip() != args.model]

    if not args.input.exists():
        print(f"Fichier introuvable : {args.input}\n"
              "Lancez d'abord : python src/data/prepare_reviews_for_llm.py",
              file=sys.stderr)
        return 1
    if not args.dry_run and not args.api_key:
        print("OPENROUTER_API_KEY manquante (fichier .env ou variable "
              "d'environnement).", file=sys.stderr)
        return 1

    # --- Point de départ -------------------------------------------------
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_valid = (
        checkpoint.get("input") == str(args.input) and args.output.exists()
    )

    if args.overwrite:
        start_line = args.start_line or 1
        if args.output.exists():
            args.output.unlink()
        checkpoint = {}
        checkpoint_valid = False
    elif args.start_line is not None:
        start_line = args.start_line
        if start_line > 1 and not checkpoint_valid:
            print("Reprise demandée mais aucun résultat précédent n'existe "
                  f"({args.output}). Utilisez --overwrite ou --start-line 1.",
                  file=sys.stderr)
            return 1
        already_done = int(checkpoint.get("last_completed_line", 0))
        if start_line <= already_done:
            print(f"Attention : les lignes {start_line} à {already_done} ont déjà "
                  "été traitées, elles seront dupliquées dans le CSV de sortie.",
                  file=sys.stderr)
    elif checkpoint_valid:
        start_line = int(checkpoint.get("last_completed_line", 0)) + 1
    else:
        start_line = 1

    total_rows = count_data_rows(args.input)
    if start_line > total_rows:
        print(f"Toutes les lignes sont déjà traitées ({total_rows} lignes). "
              "Rien à faire.")
        return 0

    to_process = total_rows - start_line + 1
    if args.limit is not None:
        to_process = min(to_process, args.limit)

    print(f"Modele        : {args.model}")
    if args.fallback_models:
        print(f"Secours       : {', '.join(args.fallback_models)}")
    print(f"Entree        : {args.input}")
    print(f"Sortie        : {args.output}")
    print(f"Lignes totales: {total_rows}")
    print(f"Depart ligne  : {start_line}"
          + ("  (reprise automatique)"
             if checkpoint_valid and args.start_line is None else ""))
    print(f"A traiter     : {to_process} lignes")
    print()

    classifier = SentimentClassifier(
        api_key=args.api_key or "",
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        max_chars=args.max_chars,
        verbose=not args.quiet,
        fallback_models=args.fallback_models,
    )

    # --- Dry run ---------------------------------------------------------
    if args.dry_run:
        first = next(iter_parts(args.input, start_line,
                                min(to_process, args.batch_size)))
        records = to_records(first[0])
        print("----- PROMPT (aperçu du 1er lot) -----")
        print(classifier.build_prompt(records))
        print("--------------------------------------")
        cost = estimate_cost_per_row(records, args.model, args.batch_size)
        if cost is not None:
            print(f"Cout estime pour {to_process} lignes : ~{cost * to_process:.2f} $")
        else:
            print("Prix inconnu pour ce modèle : estimation indisponible.")
        return 0

    # --- Boucle principale ----------------------------------------------
    done = 0
    errors = 0
    last_line = start_line - 1
    started = time.time()
    interrupted = False

    base_rows = int(checkpoint.get("rows_done_total", 0))
    base_errors = int(checkpoint.get("errors_total", 0))
    base_cost = float(checkpoint.get("total_cost_usd", 0.0))

    def process_block(block: pd.DataFrame) -> tuple[list[dict], bool]:
        """Classifie un bloc. Retourne (lignes, interrompu).

        En cas de Ctrl+C, les lots déjà terminés sont conservés : ils sont
        quand même écrits dans le CSV (rien n'est perdu).
        """
        records = to_records(block)
        batches = [records[i:i + args.batch_size]
                   for i in range(0, len(records), args.batch_size)]

        rows: list[dict] = []
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = []
        try:
            futures = [pool.submit(classifier.classify_batch, b) for b in batches]
            for batch, future in zip(batches, futures):
                labels = future.result()
                rows.extend({**rec, "Sentiment": label}
                            for rec, label in zip(batch, labels))
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            return rows, True
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return rows, False

    def flush(rows: list[dict]) -> None:
        """Écrit les lignes et met à jour le checkpoint (appelé très souvent)."""
        nonlocal done, errors, last_line, checkpoint
        if not rows:
            return
        append_rows(args.output, rows)
        done += len(rows)
        errors += sum(1 for r in rows if r["Sentiment"] == -1)
        last_line += len(rows)

        checkpoint = {
            "input": str(args.input),
            "output": str(args.output),
            "model": args.model,
            "last_completed_line": last_line,
            "rows_done_total": base_rows + done,
            "errors_total": base_errors + errors,
            "total_cost_usd": round(base_cost + classifier.total_cost, 6),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_checkpoint(args.checkpoint, checkpoint)

        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        eta = (to_process - done) / rate if rate else 0
        print(f"[{done}/{to_process}] ligne {last_line} | "
              f"{rate:.1f} avis/s | ETA {eta / 60:.1f} min | "
              f"-1: {errors} | {classifier.total_cost:.4f} $", flush=True)

    try:
        for part, _ in iter_parts(args.input, start_line, to_process):
            # Sauvegarde tous les `--save-every` avis : une coupure ne fait
            # perdre au pire qu'un bloc, jamais tout le fichier.
            for offset in range(0, len(part), args.save_every):
                block = part.iloc[offset:offset + args.save_every]
                rows, stop = process_block(block)
                flush(rows)
                if stop:
                    raise KeyboardInterrupt

    except FatalAPIError as exc:
        print(f"\nArret : {exc}", file=sys.stderr)
        interrupted = True
    except KeyboardInterrupt:
        print("\nInterrompu par l'utilisateur.", file=sys.stderr)
        interrupted = True

    # --- Résumé ----------------------------------------------------------
    elapsed = time.time() - started
    print()
    print("=" * 60)
    print(f"Lignes traitees dans ce run : {done}")
    print(f"Derniere ligne terminee     : {last_line}")
    print(f"Reprise possible a la ligne : {last_line + 1}")
    print(f"Exceptions (-1)             : {errors}")
    print(f"Duree                       : {elapsed / 60:.1f} min")
    print(f"Cout API (ce run)           : {classifier.total_cost:.4f} $")
    if classifier.served_models:
        served = ", ".join(f"{m} ({n})"
                           for m, n in sorted(classifier.served_models.items()))
        print(f"Modeles utilises            : {served}")
    print(f"Checkpoint                  : {args.checkpoint}")
    print("=" * 60)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
