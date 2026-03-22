import json
import os
import time
import re
from groq import Groq
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
ERROR_FILE = "tweets_errors.json"
OUTPUT_FILE = "tweets_sentimento.json"

MODEL = "llama-3.1-8b-instant"

DELAY = 2  # segundos entre requests (evita bloqueio)

# ======================================
# PROMPT
# ======================================
SYSTEM_PROMPT = """
Classifique o sentimento do tweet como:
POSITIVO, NEGATIVO ou NEUTRO.

Regras:
- opinião positiva → POSITIVO
- crítica → NEGATIVO
- informação → NEUTRO

Responda APENAS uma palavra.
"""

# ======================================
# CLIENT
# ======================================
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("Defina GROQ_API_KEY")
    return Groq(api_key=api_key)

# ======================================
# LOAD / SAVE
# ======================================
def load_json_safe(file):
    if not os.path.exists(file):
        return []
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_json_safe(file, data):
    temp = file + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, file)

# ======================================
# CLASSIFY SINGLE (ULTRA SEGURO)
# ======================================
def classify_one(client, text):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:300]}
            ]
        )

        result = response.choices[0].message.content.strip().upper()

        if "POSITIVO" in result:
            return "POSITIVO"
        elif "NEGATIVO" in result:
            return "NEGATIVO"
        else:
            return "NEUTRO"

    except:
        return None

# ======================================
# MAIN
# ======================================
def main():
    client = get_client()

    errors = load_json_safe(ERROR_FILE)
    results = load_json_safe(OUTPUT_FILE)

    processed_ids = {t.get("tweet_id") for t in results}

    # remove duplicados
    errors = [t for t in errors if t.get("tweet_id") not in processed_ids]

    print(f"🔁 Reprocessando {len(errors)} tweets")

    still_failed = []
    success_count = 0

    for tweet in tqdm(errors):
        sentiment = classify_one(client, tweet.get("content", ""))

        if sentiment:
            tweet["sentiment"] = sentiment
            results.append(tweet)
            success_count += 1

            # 🔥 salva a cada sucesso
            save_json_safe(OUTPUT_FILE, results)

        else:
            still_failed.append(tweet)

        time.sleep(DELAY)

    # salva erros restantes
    save_json_safe(ERROR_FILE, still_failed)

    print("\n✅ FINALIZADO")
    print(f"✔ Sucesso: {success_count}")
    print(f"❌ Ainda com erro: {len(still_failed)}")

# ======================================
if __name__ == "__main__":
    main()