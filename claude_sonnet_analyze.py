import anthropic
import json
import os
import time
from tqdm import tqdm

# ======================================
# CONFIG CLAUDE
# ======================================
ANTHROPIC_API_KEY = "sua chave aqui"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

INPUT_FILE = "tweets.json"
OUTPUT_FILE = "classified_sentiment_claude.json"

# O modelo 'haiku' é o melhor custo-benefício para 15k tweets.
# Se quiser o máximo de precisão, use 'claude-3-5-sonnet-20240620'
MODEL_NAME = "claude-3-haiku-20240307"

# ======================================
# O MESMO PROMPT (RIGOROSAMENTE IGUAL)
# ======================================
def build_prompt(text):
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

Texto: "{text}"

Responda com APENAS UMA palavra:
POSITIVO, NEGATIVO ou NEUTRO.

Não explique.
Não justifique.
Não adicione pontuação.
Não escreva nada além da resposta.
[/INST]
"""

def normalize_response(response_text: str) -> str:
    response = response_text.upper().strip()
    for char in [".", ":", "-", "\n"]:
        response = response.replace(char, "")
    word = response.split(" ")[0]
    
    if "POSITIVO" in word: return "POSITIVO"
    if "NEGATIVO" in word: return "NEGATIVO"
    return "NEUTRO"

# ======================================
# EXECUÇÃO
# ======================================
def main():
    if not os.path.exists(INPUT_FILE):
        print("❌ Erro: Arquivo de entrada não encontrado.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tweets = json.load(f)

    # Lógica de checkpoint para não gastar dinheiro/tokens à toa
    results = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)

    processed_ids = {t.get("tweet_id") for t in results}
    to_process = [t for t in tweets if t.get("tweet_id") not in processed_ids]

    print(f"🚀 Iniciando Claude para {len(to_process)} tweets...")

    for i, tweet in enumerate(tqdm(to_process)):
        text = tweet.get("content", "")[:300]
        
        try:
            prompt = build_prompt(text)
            
            # Chamada da API Anthropic
            message = client.messages.create(
                model=MODEL_NAME,
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Pegando apenas o texto da resposta
            raw_response = message.content[0].text
            tweet["sentiment_claude"] = normalize_response(raw_response)
            
        except Exception as e:
            print(f"\nErro no tweet {tweet.get('tweet_id')}: {e}")
            time.sleep(5) # Pausa em caso de erro de rede ou limite
            tweet["sentiment_claude"] = "NEUTRO"

        results.append(tweet)

        # O Claude tem limites por minuto (RPM). 
        # Para 15k tweets, uma pausa de 1 ou 2 segundos costuma ser suficiente.
        time.sleep(1) 

        if i % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("✅ Processamento Claude finalizado!")

if __name__ == "__main__":
    main()