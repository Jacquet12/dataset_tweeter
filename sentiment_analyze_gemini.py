import google.generativeai as genai
import json
import os
import time
from tqdm import tqdm

# ======================================
# CONFIG GEMINI
# ======================================
# Chave extraída da sua imagem anterior
GOOGLE_API_KEY = "sua chave aqui" 
genai.configure(api_key=GOOGLE_API_KEY)

INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_gemini.json"
ERROR_FILE = "tweets_errors_gemini.json"

# Modelo Flash: ideal para grandes volumes e gratuito
model = genai.GenerativeModel('gemini-1.5-flash')

# ======================================
# PROMPT AJUSTADO (SEM VIÉS PARA NEUTRO)
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
# NORMALIZAÇÃO
# ======================================
def normalize_response(response: str) -> str:
    response = response.upper().strip()
    for char in [".", ":", "-", "\n", "!"]:
        response = response.replace(char, "")
    
    # Pega apenas a primeira palavra caso o modelo fale demais
    response = response.split(" ")[0]

    if "POSITIVO" in response: return "POSITIVO"
    elif "NEGATIVO" in response: return "NEGATIVO"
    else: return "NEUTRO"

# ======================================
# CLASSIFY GEMINI
# ======================================
def classify_tweet(text):
    prompt = build_prompt(text)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0, # Garantir determinismo acadêmico
                max_output_tokens=10 
            )
        )
        return normalize_response(response.text)
    except Exception as e:
        # Erro de Rate Limit (429) ou conexão
        if "429" in str(e):
            time.sleep(15)
        return "NEUTRO"

# ======================================
# MAIN
# ======================================
def main():
    # Carregamento seguro do arquivo de entrada
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Arquivo {INPUT_FILE} não encontrado!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    # Carregamento de progresso anterior (Checkpoint)
    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except:
            results = []

    # Filtrar o que já foi processado
    processed_ids = {str(t.get("tweet_id")) for t in results}
    tweets_to_process = [t for t in tweets if str(t.get("tweet_id")) not in processed_ids]

    print(f"🚀 Gemini processando: {len(tweets_to_process)} tweets")

    for i, tweet in enumerate(tqdm(tweets_to_process)):
        text = tweet.get("content", "")[:300] # Aumentado para 300 para melhor contexto

        if not text.strip():
            tweet["sentiment_gemini"] = "NEUTRO"
        else:
            sentiment = classify_tweet(text)
            tweet["sentiment_gemini"] = sentiment
            
        results.append(tweet)

        # PAUSA OBRIGATÓRIA (Limite Grátis: 15 RPM)
        time.sleep(4.2)

        # Salva a cada 20 tweets para não perder progresso
        if i % 20 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Salvamento final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ FINALIZADO COM SUCESSO!")

if __name__ == "__main__":
    main()