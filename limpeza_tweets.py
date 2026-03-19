import json
import time
import os
from groq import Groq

# =========================================================
# CONFIGURAÇÕES - TCC (MODO TURBO)
# =========================================================

client = Groq(api_key="usa chave aqui")

INPUT_FILE = "tweets_nitter.json"
OUTPUT_FILE = "tweets_filtrados.json"

# Modelo de alta velocidade
MODELO = "llama-3.1-8b-instant"

PROMPT_SISTEMA = "Responda apenas RELEVANTE ou NAO_RELEVANTE. O tweet fala sobre ferramentas de IA (ChatGPT, Gemini, Copilot, DeepSeek, etc)? Tweet: "

def classificar(texto):
    """Função para chamar a API com tratamento de cota (429)"""
    try:
        completion = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": PROMPT_SISTEMA + texto}],
            temperature=0,
        )
        return completion.choices[0].message.content.strip().upper()
    except Exception as e:
        if "429" in str(e):
            return "ESPERAR"
        print(f"\nErro na Groq: {e}")
        return "ERRO"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo {INPUT_FILE} não encontrado!")
        return

    # 1. Carregar dados originais
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dados_originais = json.load(f)

    # 2. Sistema de Continuidade (Checkpoint)
    resultados = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                resultados = json.load(f)
            print(f"--- Retomando do tweet {len(resultados) + 1} ---")
        except:
            print("--- Iniciando novo processamento ---")

    total_total = len(dados_originais)
    print(f"MODO TURBO: {MODELO} com delay de 0.8s.")

    # 3. Loop de Processamento
    for i in range(len(resultados), total_total):
        tweet = dados_originais[i]
        texto = tweet.get("content", "")

        if not texto:
            tweet["relevante"] = "NAO_RELEVANTE"
            resultados.append(tweet)
            continue

        # Tenta classificar até ter sucesso (em caso de 429)
        sucesso = False
        while not sucesso:
            status = classificar(texto)

            if status == "ESPERAR":
                print(f"\n[!] Cota atingida. Pausando 45s para liberar...")
                time.sleep(45)
                continue # Tenta o mesmo tweet novamente
            
            if "ERRO" in status:
                print(f"\n[!] Parada por erro crítico no tweet {i+1}. Salvando...")
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(resultados, f, ensure_ascii=False, indent=2)
                return

            # Se chegou aqui, deu certo
            # Limpeza rápida da resposta da IA
            final_status = "RELEVANTE" if "RELEVANTE" in status and "NAO" not in status else "NAO_RELEVANTE"
            
            tweet["relevante"] = final_status
            resultados.append(tweet)
            sucesso = True
            
            print(f"[{i+1}/{total_total}] Classificado: {final_status}", end="\r")

        # Salva o arquivo a cada 20 tweets para não perder tempo com disco
        if (i + 1) % 20 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(resultados, f, ensure_ascii=False, indent=2)
        
        # Delay reduzido para 0.8s (Agressivo)
        time.sleep(0.8)

    # 4. Salvamento Final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nSUCESSO! Total processado: {len(resultados)}")

if __name__ == "__main__":
    main()