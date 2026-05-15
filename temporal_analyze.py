import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# ==============================
# CONFIG
# ==============================
ARQUIVO = "dataset_final_v2.json"

PASTA_SAIDA = "graficos_temporais"
os.makedirs(PASTA_SAIDA, exist_ok=True)

cores = {
    "POSITIVO": "#1f77b4",  # azul
    "NEUTRO": "#7f7f7f",    # cinza
    "NEGATIVO": "#d62728"   # vermelho
}

# ==============================
# LOAD DATA
# ==============================
with open(ARQUIVO, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# ==============================
# LIMPEZA
# ==============================
df = df.dropna(subset=["days_after_event"])

df["days_after_event"] = df["days_after_event"].astype(int)

# ==============================
# TEMPO RELATIVO → SEMANAS
# ==============================
df["week"] = df["days_after_event"] // 7

# limitar até 90 dias (~13 semanas)
df = df[df["week"] <= 12]

# ==============================
# DEFINIR EVENTOS
# ==============================
# Ajuste conforme os nomes reais do seu dataset

eventos = {
    "eventos_iniciais": [
        "initial"
    ],
    "eventos_recentes": [
        "latest"
    ]
}

# ==============================
# FUNÇÃO PARA GERAR GRÁFICOS
# ==============================
def gerar_grafico(df_filtrado, titulo, nome_arquivo):

    grouped = (
        df_filtrado
        .groupby(["week", "final_sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    # porcentagem
    grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # ordem fixa
    ordem = ["POSITIVO", "NEUTRO", "NEGATIVO"]

    grouped_pct = grouped_pct[
        [c for c in ordem if c in grouped_pct.columns]
    ]

    # ==============================
    # PLOT
    # ==============================
    plt.style.use("default")

    plt.figure(figsize=(10, 5))

    for sentiment in grouped_pct.columns:
        plt.plot(
            grouped_pct.index,
            grouped_pct[sentiment],
            marker='o',
            linewidth=2,
            label=sentiment,
            color=cores[sentiment]
        )

    # ==============================
    # ESTILO
    # ==============================
    plt.title(titulo)
    plt.xlabel("Semanas após o evento")
    plt.ylabel("Porcentagem (%)")

    plt.xticks(range(0, 13))
    plt.yticks(range(0, 101, 10))

    plt.ylim(0, 100)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.legend(title="Sentimento")

    plt.tight_layout()

    # ==============================
    # SAVE
    # ==============================
    caminho = os.path.join(PASTA_SAIDA, nome_arquivo)

    plt.savefig(caminho, dpi=300)

    plt.close()

    print(f"Gráfico salvo em: {caminho}")

# ==============================
# GERAR GRÁFICOS
# ==============================

# Eventos iniciais
df_iniciais = df[df["event"].isin(eventos["eventos_iniciais"])]

gerar_grafico(
    df_iniciais,
    "Evolução Temporal dos Sentimentos - Eventos Iniciais",
    "evolucao_eventos_iniciais.png"
)

# Eventos recentes
df_recentes = df[df["event"].isin(eventos["eventos_recentes"])]

gerar_grafico(
    df_recentes,
    "Evolução Temporal dos Sentimentos - Eventos Recentes",
    "evolucao_eventos_recentes.png"
)

print("\nTodos os gráficos foram gerados com sucesso.")