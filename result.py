import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- 1. CONFIGURAÇÕES E CRIAÇÃO DA PASTA ---
plt.style.use('dark_background')
cores = {'POSITIVO': '#00f2ff', 'NEUTRO': '#7d7d7d', 'NEGATIVO': '#ff4b4b'}
fundo_card = '#0d1117'
ARQUIVO_JSON = 'dataset_final_v2.json'
PASTA_SAIDA = 'resultados_tcc'

# Cria a pasta se ela não existir
if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)
    print(f"📁 Pasta '{PASTA_SAIDA}' criada com sucesso!")

def carregar_dados():
    with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['event'] = df['event'].replace({'launch_initial': 'Lançamento', 'latest_version': 'Versão Atual'})
    return df

# --- 2. GRÁFICO GLOBAL (DONUT) ---
def gerar_donut_global(df):
    counts = df['final_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=fundo_card)
    
    wedges, texts, autotexts = ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                                      startangle=140, colors=[cores[s] for s in counts.index],
                                      pctdistance=0.85, explode=[0.05]*len(counts))
    
    centre_circle = plt.Circle((0,0), 0.70, fc=fundo_card)
    fig.gca().add_artist(centre_circle)
    plt.title('DISTRIBUIÇÃO GLOBAL DE SENTIMENTOS', fontsize=16, fontweight='bold', pad=20, color=cores['POSITIVO'])
    
    caminho = os.path.join(PASTA_SAIDA, '01_global_donut.png')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()

# --- 3. EVOLUÇÃO POR PRODUTO (GPT, GEMINI, COPILOT) ---
def gerar_evolucao_produtos(df):
    produtos = ['gpt', 'gemini', 'copilot']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=fundo_card)
    
    for i, prod in enumerate(produtos):
        df_p = df[df['model'] == prod]
        data = df_p.groupby(['event', 'final_sentiment']).size().unstack(fill_value=0)
        data_pct = data.div(data.sum(axis=1), axis=0) * 100
        
        ordem = [c for c in ['NEGATIVO', 'NEUTRO', 'POSITIVO'] if c in data_pct.columns]
        data_pct[ordem].plot(kind='bar', ax=axes[i], color=[cores[c] for c in ordem], width=0.8)
        
        axes[i].set_title(prod.upper(), fontsize=16, fontweight='bold', color=cores['POSITIVO'])
        axes[i].set_ylim(0, 100)
        axes[i].set_xticklabels(['Lançamento', 'Versão Atual'], rotation=0)
        
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_height():.0f}%', (p.get_x() + p.get_width()/2., p.get_height()+1), 
                             ha='center', color='white', weight='bold')

    plt.tight_layout()
    caminho = os.path.join(PASTA_SAIDA, '02_evolucao_produtos.png')
    plt.savefig(caminho, dpi=300)
    plt.close()

# --- 4. EVOLUÇÃO TEMPORAL (ÁREA) ---
def gerar_temporal(df):
    temporal = df.groupby([df['date'].dt.to_period('W'), 'final_sentiment']).size().unstack(fill_value=0)
    temporal_pct = temporal.div(temporal.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=fundo_card)
    # Garante que as cores sigam o padrão Neg/Neu/Pos
    ordem_cores = [cores.get(c) for c in temporal_pct.columns]
    
    temporal_pct.plot(kind='area', stacked=True, ax=ax, color=ordem_cores, alpha=0.6)
    
    plt.title('FLUXO TEMPORAL DE SENTIMENTOS (SEMANAL)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Porcentagem (%)')
    
    caminho = os.path.join(PASTA_SAIDA, '03_fluxo_temporal.png')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()

# --- 5. CONCORDÂNCIA DO ENSEMBLE ---
def gerar_concordancia(df):
    auxiliares = ['llama', 'phi', 'deepseek']
    taxas = [round((df[mod] == df['mistral']).mean() * 100, 2) for mod in auxiliares]
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=fundo_card)
    barras = ax.bar([a.upper() for a in auxiliares], taxas, color=['#444444', cores['POSITIVO'], '#444444'], width=0.6)
    
    plt.ylim(0, 100)
    plt.title('TAXA DE CONCORDÂNCIA COM O MISTRAL', fontsize=14, pad=20)
    for b in barras:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f'{b.get_height()}%', ha='center', weight='bold')
    
    caminho = os.path.join(PASTA_SAIDA, '04_concordancia_ensemble.png')
    plt.savefig(caminho, dpi=300)
    plt.close()

# --- EXECUÇÃO ---
try:
    print("🚀 Iniciando processamento do dataset...")
    df_main = carregar_dados()
    
    print("🎨 Gerando gráficos modernos...")
    gerar_donut_global(df_main)
    gerar_evolucao_produtos(df_main)
    gerar_temporal(df_main)
    gerar_concordancia(df_main)
    
    print(f"\n✅ Sucesso! Verifique a pasta '{PASTA_SAIDA}' para ver os resultados.")
except Exception as e:
    print(f"❌ Erro: {e}")