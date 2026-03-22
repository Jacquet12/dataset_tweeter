import json
import os
import time
import re
from groq import Groq
from tqdm import tqdm

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "tweets_sentimento.json"
ERROR_FILE = "tweets_errors.json"

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]

BATCH_SIZE = 20
REQUESTS_PER_MINUTE = 30
SECONDS_PER_REQUEST = 60 / REQUESTS_PER_MINUTE
MAX_RETRIES = 3

# ======================================
# PROMPT ENGINEERING AVANÇADO
# ======================================
SYSTEM_PROMPT = """
Você é um especialista em análise de sentimentos para pesquisa acadêmica sobre Inteligência Artificial.

Seu objetivo é classificar tweets relacionados a IA.

Classifique cada tweet como:
- POSITIVO
- NEGATIVO
- NEUTRO

CRITÉRIOS DETALHADOS:

1. POSITIVO:
- entusiasmo com IA
- elogios a ferramentas (ChatGPT, Gemini, etc.)
- visão otimista sobre impacto

2. NEGATIVO:
- medo (ex: perda de empregos)
- críticas ou frustração
- visão pessimista

3. NEUTRO:
- notícias ou anúncios
- descrição factual sem opinião
- compartilhamento de link sem emoção

REGRAS IMPORTANTES:
- Tweets curtos com link geralmente são NEUTROS
- Ironia deve ser interpretada corretamente
- Emojis podem indicar sentimento
- Se não houver emoção clara → NEUTRO

SAÍDA:
- Responda APENAS JSON válido
- NÃO explique nada
- NÃO use markdown

Formato:
[
  {
    "item_index": 0,
    "sentiment": "POSITIVO"
  }
]
"""

# ======================================
# CLIENT
# ======================================
def get_client():
    api_key = os.getenv("Defina GROQ_API_KEY  aqui")
    if not api_key:
        raise Exception("Defina GROQ_API_KEY")
    return Groq(api_key=api_key)

# ======================================
# SAFE LOAD
# ======================================
def load_json_safe(file):
    if not os.path.exists(file):
        return []
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

# ======================================
# SAFE SAVE
# ======================================
def save_json_safe(file, data):
    temp_file = file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, file)

# ======================================
# PARSER
# ======================================
def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)

# ======================================
# CLASSIFY
# ======================================
def classify_batch(client, batch):
    payload = [
        {"item_index": i, "text": t.get("content", "")[:300]}
        for i, t in enumerate(batch)
    ]

    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                    ]
                )
                return extract_json(response.choices[0].message.content)

            except Exception as e:
                print(f"⚠️ {model} tentativa {attempt+1} falhou")
                time.sleep(2)

    raise Exception("Todos modelos falharam")

# ======================================
# MAIN
# ======================================
def main():
    client = get_client()

    tweets = load_json_safe(INPUT_FILE)
    existing = load_json_safe(OUTPUT_FILE)
    errors = load_json_safe(ERROR_FILE)

    processed_ids = {t.get("tweet_id") for t in existing}

    tweets = [t for t in tweets if t.get("tweet_id") not in processed_ids]

    results = existing.copy()

    batches = [tweets[i:i+BATCH_SIZE] for i in range(0, len(tweets), BATCH_SIZE)]

    print(f"🚀 Restantes: {len(tweets)} tweets em {len(batches)} lotes")

    last_request = 0

    for batch in tqdm(batches):
        elapsed = time.time() - last_request
        if elapsed < SECONDS_PER_REQUEST:
            time.sleep(SECONDS_PER_REQUEST - elapsed)

        try:
            response = classify_batch(client, batch)
            last_request = time.time()

            new_items = []

            for item in response:
                idx = item.get("item_index")

                if idx is None or idx >= len(batch):
                    continue

                sentiment = item.get("sentiment", "NEUTRO")

                tweet = batch[idx]

                # 🔥 mantém seu formato original
                tweet["sentiment"] = sentiment

                new_items.append(tweet)

            results.extend(new_items)

            # 🔥 salva a cada lote
            save_json_safe(OUTPUT_FILE, results)

        except Exception as e:
            print(f"\n❌ Erro no lote: {e}")
            errors.extend(batch)
            save_json_safe(ERROR_FILE, errors)
            continue

    print("✅ FINALIZADO")

# ======================================
if __name__ == "__main__":
    main()