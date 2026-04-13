import json
import os
import time
import requests
from tqdm import tqdm

# ======================================
# CONFIGURAÇÃO
# ======================================
INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_deepseek_v1.json"

# Modelo leve e rápido que NÃO TRAVA no Hugging Face
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
HF_TOKEN = "seu token aqui"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Intervalo curto para o contador andar rápido
SECONDS_PER_REQUEST = 1.0 
SAVE_INTERVAL = 20

# ======================================
# SEU PROMPT PADRÃO (METODOLOGIA TCC)
# ======================================
def build_prompt(text):
    return f"""<|user|>
[INST]
Você é um classificador de sentimentos.
Sua tarefa é classificar o sentimento do texto em uma única categoria: POSITIVO, NEGATIVO ou NEUTRO.

Definições:
- POSITIVO: o texto expressa opinião favorável, entusiasmo, satisfação ou expectativa positiva.
- NEGATIVO: o texto expressa crítica, insatisfação, preocupação, medo ou expectativa negativa.
- NEUTRO: o texto é informativo, descritivo, ambíguo ou não expressa claramente uma opinião ou emoção.

Diretrizes:
- Considere o significado geral do texto, tom da linguagem e contexto implícito.
- Emojis podem indicar sentimento. Interprete ironia quando possível.
- Se houver ambiguidade clara, classifique como NEUTRO.

Texto: "{text}"

Responda com APENAS UMA palavra: POSITIVO, NEGATIVO ou NEUTRO.
Não explique. Não justifique.
[/INST]<|end|>
<|assistant|>
<|thought|>
"""

def normalize_response(response_text):
    # Pega apenas o resultado após o raciocínio do modelo
    res = response_text.split("</thought>")[-1] if "</thought>" in response_text else response_text
    res = res.upper().strip()
    for c in [".", ":", "-", "\n", "!", "[", "]", "/", "*"]: res = res.replace(c, "")
    
    word = res.split(" ")[0]
    if "POSITIVO" in word: return "POSITIVO"
    if "NEGATIVO" in word: return "NEGATIVO"
    return "NEUTRO"

def classify_tweet(text):
    payload = {
        "inputs": build_prompt(text),
        "parameters": {"max_new_tokens": 15, "return_full_text": False, "temperature": 0.5}
    }
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                raw_out = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
                return normalize_response(raw_out)
            elif response.status_code == 503: # Modelo carregando
                time.sleep(15)
                continue
            elif response.status_code == 429: # Rate limit
                time.sleep(60)
                continue
            else:
                time.sleep(10)
                continue
        except:
            time.sleep(5)

def main():
    if not os.path.exists(INPUT_FILE): return

    # Checkpoint Seguro
    results = []
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except: results = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    processed_ids = {str(t.get("tweet_id")).split("#")[0] for t in results if t.get("tweet_id")}
    to_process = [t for t in tweets if str(t.get("tweet_id")).split("#")[0] not in processed_ids]

    print(f"📊 Processados: {len(results)} | Faltam: {len(to_process)}")
    print(f"🚀 Iniciando {MODEL_ID} no Hugging Face...")

    last_req = 0
    for i, tweet in enumerate(tqdm(to_process)):
        now = time.time()
        if now - last_req < SECONDS_PER_REQUEST:
            time.sleep(SECONDS_PER_REQUEST - (now - last_req))

        content = tweet.get("content", "")[:300]
        tweet["sentiment_deepseek_v1"] = classify_tweet(content) if content.strip() else "NEUTRO"
        
        results.append(tweet)
        last_req = time.time()

        if i % SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE + ".tmp", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            os.replace(OUTPUT_FILE + ".tmp", OUTPUT_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("✅ FINALIZADO!")

if __name__ == "__main__":
    main()