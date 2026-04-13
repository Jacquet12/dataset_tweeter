import json
import os
import time
import requests
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_deepseek.json"

MODEL_NAME = "llama3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SAVE_INTERVAL = 20
SLEEP_BETWEEN_REQUESTS = 0.1

# ======================================
# PROMPT LEVE (OTIMIZADO)
# ======================================
def build_prompt(text):
    return f"""
Classifique o sentimento do texto como:
POSITIVO, NEGATIVO ou NEUTRO.

Texto: "{text}"

Responda com apenas uma palavra:
POSITIVO, NEGATIVO ou NEUTRO.
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
# CLASSIFY
# ======================================
def classify(text):
    payload = {
        "model": MODEL_NAME,
        "prompt": build_prompt(text),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 5  # força resposta curta
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
                print("❌ resposta inválida:", data)
                time.sleep(1)
                continue

            return normalize_response(data["response"])

        except Exception as e:
            print(f"⚠️ tentativa {attempt+1} falhou:", e)
            time.sleep(1)

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
        content = tweet.get("content", "")[:100]  # 🔥 menor texto

        if not content.strip():
            sentiment = "NEUTRO"
        else:
            sentiment = classify(content)

        tweet["sentiment_deepseek"] = sentiment
        results.append(tweet)

        if i % SAVE_INTERVAL == 0:
            save_json_safe(OUTPUT_FILE, results)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    save_json_safe(OUTPUT_FILE, results)
    print("✅ FINALIZADO")

if __name__ == "__main__":
    main()