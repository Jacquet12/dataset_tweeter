import json
import os

ENTRADA = "tweets_filtrados.json"
SAIDA_FINAL = "tweets_TCC_corrigidos.json"

# Lista de termos-chave para o resgate (em minúsculo)
TERMOS_RESGATE = [
    "chatgpt", "gpt", "openai", "gemini", "copilot", 
    "ia", "ai", "inteligencia artificial", "inteligência artificial",
    "deepseek", "claude", "midjourney", "dall-e"
]

def corrigir_base_dados():
    if not os.path.exists(ENTRADA):
        print(f"Arquivo {ENTRADA} não encontrado!")
        return

    with open(ENTRADA, "r", encoding="utf-8") as f:
        dados = json.load(f)

    base_final = []
    recuperados = 0

    print("Iniciando correção de falsos negativos...")

    for tweet in dados:
        # Pega o texto e transforma em minúsculo apenas para a verificação
        texto_comparacao = tweet.get("content", "").lower()
        status_ia = tweet.get("relevante", "")

        # Se a IA descartou, verificamos se as palavras-chave estão lá
        if status_ia == "NAO_RELEVANTE":
            # O 'any' verifica se qualquer um dos termos está no texto (independente de ser MAIÚSCULO no original)
            if any(termo in texto_comparacao for termo in TERMOS_RESGATE):
                tweet["relevante"] = "RELEVANTE" # Muda o status no dicionário
                base_final.append(tweet)
                recuperados += 1
            else:
                # Se realmente não tem as palavras, fica de fora
                pass
        else:
            # Se a IA já tinha marcado como RELEVANTE, mantemos na base
            base_final.append(tweet)

    # Salva o arquivo final com todos os relevantes (originais + recuperados)
    with open(SAIDA_FINAL, "w", encoding="utf-8") as f:
        json.dump(base_final, f, ensure_ascii=False, indent=2)

    print("-" * 40)
    print(f"RESULTADO DA CORREÇÃO:")
    print(f"Tweets recuperados (que a IA errou): {recuperados}")
    print(f"Total de tweets relevantes para o TCC: {len(base_final)}")
    print(f"Arquivo gerado: {SAIDA_FINAL}")
    print("-" * 40)

if __name__ == "__main__":
    corrigir_base_dados()