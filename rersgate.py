import json

# =========================
# CONFIG
# =========================
FILES = {
    "mistral": "classified_sentiment_mistral_other.json",
    "llama": "classified_sentiment_llama.json",
    "phi": "classified_sentiment_phi4.json",
    "deepseek": "classified_sentiment_deepseek.json"
}

OUTPUT_FILE = "dataset_final_v2.json"

# =========================
# FUNÇÕES
# =========================
def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_id(tweet_id):
    return tweet_id.split("#")[0] if tweet_id else None

def normalize_sentiment(s):
    if not s:
        return "NEUTRO"
    s = str(s).upper()

    if "POS" in s:
        return "POSITIVO"
    if "NEG" in s:
        return "NEGATIVO"

    return "NEUTRO"

# =========================
# LOAD
# =========================
data = {k: load_json(v) for k, v in FILES.items()}

# =========================
# INDEXAR POR TWEET_ID
# =========================
indexed = {}

for model, items in data.items():
    for item in items:

        tid = clean_id(item.get("tweet_id"))
        if not tid:
            continue

        # criar base
        if tid not in indexed:
            indexed[tid] = {
                "tweet_id": tid,
                "content": item.get("content"),
                "model": item.get("model"),
                "event": item.get("event"),
                "event_date": item.get("event_date"),
                "date": item.get("date"),
                "days_after_event": item.get("days_after_event"),
                "author": item.get("author"),
                "source": item.get("source"),

                # votos
                "llama": None,
                "mistral": None,
                "phi": None,
                "deepseek": None
            }

        # pegar sentimento correto (cada modelo usa campo diferente)
        sentiment = (
            item.get("sentiment")
            or item.get("sentiment_phi3")
            or item.get("sentiment_deepseek")
        )

        indexed[tid][model] = normalize_sentiment(sentiment)

# =========================
# CALCULAR VOTAÇÃO
# =========================
final_data = []

for item in indexed.values():

    mistral = item["mistral"]

    votes = 0

    for m in ["llama", "phi", "deepseek"]:
        if item[m] == mistral:
            votes += 1

    # =========================
    # CAMPOS FINAIS
    # =========================
    item["final_sentiment"] = mistral
    item["votes_for_mistral"] = votes

    # nível de concordância (inclui mistral)
    item["agreement_level"] = f"{votes + 1}/4"

    # maioria concorda?
    item["mistral_supported"] = votes >= 2

    final_data.append(item)

# =========================
# SAVE
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"✅ Dataset final gerado com sucesso: {OUTPUT_FILE}")