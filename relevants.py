import json
import os

# Configurações de arquivos
ARQUIVO_ENTRADA = "tweets_filtrados.json"
ARQUIVO_RELEVANTES = "tweets_IA_relevantes.json"
ARQUIVO_DESCARTADOS = "tweets_IA_descartados.json"

def separar_tweets():
    if not os.path.exists(ARQUIVO_ENTRADA):
        print(f"Erro: O arquivo {ARQUIVO_ENTRADA} não foi encontrado!")
        return

    print("Lendo a base de dados completa...")
    with open(ARQUIVO_ENTRADA, "r", encoding="utf-8") as f:
        dados = json.load(f)

    relevantes = []
    descartados = []

    # Separação lógica
    for tweet in dados:
        # Verifica o campo 'relevante' que a IA preencheu
        status = tweet.get("relevante", "")
        
        if status == "RELEVANTE":
            relevantes.append(tweet)
        else:
            descartados.append(tweet)

    # Salva os Relevantes
    with open(ARQUIVO_RELEVANTES, "w", encoding="utf-8") as f:
        json.dump(relevantes, f, ensure_ascii=False, indent=2)

    # Salva os Descartados
    with open(ARQUIVO_DESCARTADOS, "w", encoding="utf-8") as f:
        json.dump(descartados, f, ensure_ascii=False, indent=2)

    # Relatório final para o seu TCC
    total = len(dados)
    p_relevante = (len(relevantes) / total) * 100
    
    print("-" * 30)
    print(f"RESUMO DA LIMPEZA:")
    print(f"Total processado: {total}")
    print(f"Relevantes (IA):  {len(relevantes)} ({p_relevante:.2f}%)")
    print(f"Descartados:      {len(descartados)}")
    print("-" * 30)
    print(f"Arquivo salvo: {ARQUIVO_RELEVANTES}")

if __name__ == "__main__":
    separar_tweets()