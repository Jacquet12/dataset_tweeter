import json
import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_llama.json"
ERROR_FILE = "tweets_errors_llama.json"

MODEL = "llama3:8b"

BATCH_SIZE = 3          # equilíbrio entre velocidade e memória
MAX_WORKERS = 1         # não travar CPU
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
SAVE_INTERVAL = 20      # salva progresso frequente

# ======================================
# PROMPT (INALTERADO)
# ======================================
def build_batch_prompt(texts):
    textos_formatados = "\n".join(
        [f"{i+1}. \"{t}\"" for i, t in enumerate(texts)]
    )

    return f"""
[INST]
Você é um classificador de sentimentos.

Sua tarefa é classificar o sentimento do texto em uma única categoria:
POSITIVO, NEGATIVO ou NEUTRO.

Definições:

- POSITIVO: o texto expressa opinião favorável, entusiasmo, satisfação ou expectativa positiva.

- NEGATIVO: o texto expressa crítica, insatisfação, preocupação, medo ou expectativa negativa.

- NEUTRO: o texto é informativo, descritivo, ambíguo ou não expressa claramente uma opinião ou emoção.

Diretrizes:

- Considere o significado geral do texto.
- Considere o tom da linguagem.
- Considere o contexto implícito.
- Emojis podem indicar sentimento.
- Interprete ironia ou sarcasmo quando possível.
- Não assuma sentimento sem evidência clara no texto.
- Se houver ambiguidade ou ausência de emoção clara, classifique como NEUTRO.

Textos:
{textos_formatados}

Responda com UMA linha para cada texto no formato:

1. POSITIVO
2. NEGATIVO
3. NEUTRO

Regras obrigatórias:
- Apenas uma palavra por linha
- Sem explicação
- Sem pontuação extra
- Não escreva nada além das respostas
[/INST]
"""

# ======================================
# NORMALIZAÇÃO (INALTERADO)
# ======================================
def normalize_response(resp):
    resp = resp.upper()
    if "POSITIVO" in resp:
        return "POSITIVO"
    elif "NEGATIVO" in resp:
        return "NEGATIVO"
    return "NEUTRO"

# ======================================
# PARSE (INALTERADO)
# ======================================
def parse_batch_response(raw, batch_size):
    sentiments = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentiments.append(normalize_response(line))

    while len(sentiments) < batch_size:
        sentiments.append("NEUTRO")

    return sentiments[:batch_size]

# ======================================
# CLASSIFICAÇÃO LOCAL
# ======================================
def classify_batch_local(batch):
    texts = [t.get("content", "")[:280] for t in batch]
    prompt = build_batch_prompt(texts)

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                },
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code != 200:
                time.sleep(2)
                continue

            try:
                result = response.json()
            except:
                # fallback para stream quebrado
                lines = response.text.strip().split("\n")
                result = json.loads(lines[-1])

            if "message" not in result:
                time.sleep(2)
                continue

            raw = result["message"]["content"]

            return parse_batch_response(raw, len(batch))

        except Exception:
            time.sleep(2 ** attempt)

    return ["NEUTRO"] * len(batch)

# ======================================
# CLEAN ID
# ======================================
def clean_id(tweet_id):
    return tweet_id.split("#")[0] if tweet_id else None

# ======================================
# SAVE SAFE
# ======================================
def save_json_safe(file, data):
    with open(file + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(file + ".tmp", file)

# ======================================
# MAIN
# ======================================
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    processed_ids = {clean_id(t.get("tweet_id")) for t in results}

    tweets = [
        t for t in tweets
        if clean_id(t.get("tweet_id")) not in processed_ids
    ]

    print(f"🚀 Restantes: {len(tweets)}")

    batches = [
        tweets[i:i+BATCH_SIZE]
        for i in range(0, len(tweets), BATCH_SIZE)
    ]

    counter = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(classify_batch_local, b) for b in batches]

        for future, batch in tqdm(zip(as_completed(futures), batches), total=len(batches)):
            try:
                sentiments = future.result()

                for tweet, sentiment in zip(batch, sentiments):
                    tweet["sentiment"] = sentiment
                    results.append(tweet)

                counter += 1

                # 🔥 salva incremental
                if counter % SAVE_INTERVAL == 0:
                    save_json_safe(OUTPUT_FILE, results)

            except Exception:
                print("❌ erro no batch")

    save_json_safe(OUTPUT_FILE, results)
    print("✅ FINALIZADO")

# ======================================
if __name__ == "__main__":
    main()