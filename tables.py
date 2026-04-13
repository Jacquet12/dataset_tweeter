import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INPUT_FILE = "dataset_final.json"
OUTPUT_DIR = "tables_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD
# =========================
df = pd.read_json(INPUT_FILE)
df["final_sentiment"] = df["final_sentiment"].str.upper()

# =========================
# FUNÇÃO PARA GERAR TABELA BONITA
# =========================
def save_table_image(df_table, title, filename):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        rowLabels=df_table.index,
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title(title, fontsize=12)
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# 🥇 TABELA 1 — DISTRIBUIÇÃO
# =========================
table1 = df["final_sentiment"].value_counts().to_frame("Quantidade")
table1["Porcentagem (%)"] = (table1["Quantidade"] / len(df) * 100).round(2)

save_table_image(
    table1,
    "Distribuição de Sentimentos",
    "tabela_distribuicao.png"
)

# =========================
# 🥈 TABELA 2 — CONCORDÂNCIA
# =========================
table2 = df["agreement_level"].value_counts().sort_index().to_frame("Quantidade")
table2["Porcentagem (%)"] = (table2["Quantidade"] / len(df) * 100).round(2)

save_table_image(
    table2,
    "Nível de Concordância entre Modelos",
    "tabela_concordancia.png"
)

# =========================
# 🥉 TABELA 3 — EVENTO
# =========================
table3 = pd.crosstab(df["event"], df["final_sentiment"], normalize="index") * 100
table3 = table3.round(2)

save_table_image(
    table3,
    "Distribuição de Sentimentos por Evento (%)",
    "tabela_evento.png"
)

# =========================
# 🔥 TABELA 4 — TEMPORAL
# =========================
df_time = df.dropna(subset=["days_after_event"]).copy()

bins = [0, 15, 30, 60, 90]
labels = ["0-15", "15-30", "30-60", "60-90"]

df_time["time_bin"] = pd.cut(
    df_time["days_after_event"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

table4 = pd.crosstab(
    df_time["time_bin"],
    df_time["final_sentiment"],
    normalize="index"
) * 100

table4 = table4.round(2)

save_table_image(
    table4,
    "Evolução do Sentimento por Intervalo de Tempo (%)",
    "tabela_temporal.png"
)

# =========================
print("✅ Tabelas em imagem geradas em:", OUTPUT_DIR)