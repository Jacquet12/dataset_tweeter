import json
import os
from tqdm import tqdm
from llama_cpp import Llama

# ======================================
# CONFIG
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_mistral_other.json"
ERROR_FILE = "tweets_errors.json"

MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

SAVE_INTERVAL = 100  # menos I/O = mais rápido

# ======================================
# LOAD MODEL (OTIMIZADO PRA 8GB RAM)
# ======================================
print("🚀 Carregando Mistral GGUF...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,        # menor = mais rápido e leve
    n_threads=4,      # evita travar o sistema
    n_batch=64,       # seguro para 8GB
    use_mlock=False,
    verbose=False
)

# ======================================
# PROMPT ZERO-SHOT (ACADEMICO)
# ======================================
def build_prompt(text):
    return f"""
Você é um sistema de alta precisão para análise de sentimentos.
Sua missão é identificar a polaridade real do texto abaixo, sem inclinação para qualquer categoria.

Classifique em uma única categoria: POSITIVO, NEGATIVO ou NEUTRO.

4. Responda APENAS com a palavra da categoria, sem qualquer texto adicional ou pontuação.

Texto: "{text}"

Resposta:"""

# ======================================
# UTILS
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
    temp_file = file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, file)

# ======================================
# NORMALIZAÇÃO
# ======================================
def normalize_response(response: str) -> str:
    response = response.upper().strip()

    # remove sujeiras comuns
    for char in [".", ":", "-", "\n"]:
        response = response.replace(char, "")

    # pega só a primeira palavra
    response = response.split(" ")[0]

    if response.startswith("POSITIVO"):
        return "POSITIVO"
    elif response.startswith("NEGATIVO"):
        return "NEGATIVO"
    elif response.startswith("NEUTRO"):
        return "NEUTRO"
    else:
        return "NEUTRO"

# ======================================
# CLASSIFY
# ======================================
def classify_tweet(text):
    prompt = build_prompt(text)

    try:
        output = llm(
            prompt,
            max_tokens=5,
            temperature=0,
            stop=["\n"]
        )

        raw = output["choices"][0]["text"]
        return normalize_response(raw)

    except Exception:
        return "NEUTRO"

# ======================================
# MAIN
# ======================================
def main():
    tweets = load_json_safe(INPUT_FILE)
    existing = load_json_safe(OUTPUT_FILE)
    errors = load_json_safe(ERROR_FILE)

    processed_ids = {t.get("tweet_id") for t in existing}
    tweets = [t for t in tweets if t.get("tweet_id") not in processed_ids]

    results = existing.copy()

    print(f"🚀 Restantes: {len(tweets)} tweets")

    for i, tweet in enumerate(tqdm(tweets)):
        try:
            text = tweet.get("content", "")[:200]  # reduzido para performance

            if not text.strip():
                tweet["sentiment"] = "NEUTRO"
                results.append(tweet)
                continue

            sentiment = classify_tweet(text)

            tweet["sentiment"] = sentiment
            results.append(tweet)

            # salvamento incremental otimizado
            if i % SAVE_INTERVAL == 0 and i > 0:
                save_json_safe(OUTPUT_FILE, results)

        except Exception:
            errors.append(tweet)
            save_json_safe(ERROR_FILE, errors)

    save_json_safe(OUTPUT_FILE, results)

    print("✅ FINALIZADO")

# ======================================
if __name__ == "__main__":
    main()