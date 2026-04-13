import json
import os
import time
import requests
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_phi3.json"

MODEL_NAME = "phi3:mini"
OLLAMA_URL = "http://localhost:11434/api/generate"

SAVE_INTERVAL = 20
SLEEP_BETWEEN_REQUESTS = 0.3

# ======================================
# PROMPT MELHORADO (ANTI-NEUTRO)
# ======================================
def build_prompt(text):
    return f"""
[INST]
Você é um classificador de sentimentos.

Classifique o texto como:
POSITIVO, NEGATIVO ou NEUTRO.

IMPORTANTE:
- Escolha POSITIVO ou NEGATIVO sempre que houver qualquer indicação de opinião.
- Use NEUTRO apenas se o texto for puramente informativo e sem emoção.

Exemplos:
- "I love this" → POSITIVO
- "I hate this" → NEGATIVO
- "The sky is blue" → NEUTRO

Texto: "{text}"

Responda apenas com:
POSITIVO, NEGATIVO ou NEUTRO

Regras:
- Não explique
- Não escreva frases
- Apenas uma palavra

Resposta:
[/INST]
"""

# ======================================
# HELPERS
# ======================================
def clean_id(tweet_id):
    return str(tweet_id).split("#")[0] if tweet_id else None


def load_json_safe(file):
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        return []
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_json_safe(file, data):
    temp_file = file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, file)


def normalize_response(response: str) -> str:
    res = response.upper().strip()

    for c in [".", ":", "-", "\n", "!", "[", "]", "/"]:
        res = res.replace(c, "")

    word = res.split(" ")[0]

    if "POSITIVO" in word:
        return "POSITIVO"
    if "NEGATIVO" in word:
        return "NEGATIVO"

    return "NEUTRO"


# ======================================
# CLASSIFY (ROBUSTO)
# ======================================
def classify(text):
    payload = {
        "model": MODEL_NAME,
        "prompt": build_prompt(text),
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    for attempt in range(2):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=60
            )

            data = response.json()

            if "response" not in data:
                print("❌ Erro API:", data)
                time.sleep(2)
                continue

            return normalize_response(data["response"])

        except Exception as e:
            print(f"⚠️ tentativa {attempt+1} falhou:", e)
            time.sleep(2)

    return "NEUTRO"


# ======================================
# MAIN
# ======================================
def main():
    tweets = load_json_safe(INPUT_FILE)
    results = load_json_safe(OUTPUT_FILE)

    processed_ids = {
        clean_id(t.get("tweet_id"))
        for t in results
        if t.get("tweet_id")
    }

    to_process = [
        t for t in tweets
        if clean_id(t.get("tweet_id")) not in processed_ids
    ]

    print(f"📊 Já processados: {len(processed_ids)}")
    print(f"🚀 Restantes: {len(to_process)}")

    for i, tweet in enumerate(tqdm(to_process)):
        content = tweet.get("content", "")[:180]

        if not content.strip():
            sentiment = "NEUTRO"
        else:
            sentiment = classify(content)

        tweet["sentiment_phi3"] = sentiment
        results.append(tweet)

        # salva incremental
        if i % SAVE_INTERVAL == 0:
            save_json_safe(OUTPUT_FILE, results)

        # evita sobrecarga
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    save_json_safe(OUTPUT_FILE, results)
    print("✅ FINALIZADO")


if __name__ == "__main__":
    main()