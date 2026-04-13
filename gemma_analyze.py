import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# --- 1. CONFIGURAÇÕES TÉCNICAS (PADRÃO CIENTÍFICO) ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif', # Padrão para artigos (estilo Times New Roman)
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# Cores Sólidas (Alto contraste para P&B)
# Positivo: Cinza escuro, Neutro: Cinza claro, Negativo: Branco com borda preta
cores_cientificas = {'POSITIVO': '#333333', 'NEUTRO': '#999999', 'NEGATIVO': '#f0f0f0'}
ARQUIVO_JSON = 'dataset_final_v2.json'
PASTA_SAIDA = 'artigo_cientifico_graficos'

if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)

def carregar_dados():
    with open(ARQUIVO_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['event'] = df['event'].replace({'launch_initial': 'Launch', 'latest_version': 'Latest'})
    return df

# --- 2. GRÁFICO DE BARRAS AGRUPADAS (EVOLUÇÃO DOS PRODUTOS) ---
def gerar_barras_agrupadas(df):
    produtos = ['gpt', 'gemini', 'copilot']
    sentimentos = ['POSITIVO', 'NEUTRO', 'NEGATIVO']
    
    for prod in produtos:
        df_p = df[df['model'] == prod]
        data = df_p.groupby(['event', 'final_sentiment']).size().unstack(fill_value=0)
        data_pct = data.div(data.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(6, 4))
        # Plot com bordas pretas para clareza
        data_pct.plot(kind='bar', ax=ax, color=[cores_cientificas[s] for s in data_pct.columns], 
                      edgecolor='black', linewidth=1)
        
        ax.set_title(f'Sentiment Analysis: {prod.upper()}')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Event Phase')
        ax.set_ylim(0, 100)
        ax.legend(title='Sentiment', frameon=True)
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_SAIDA, f'fig_scientific_{prod}.pdf')) # Salva em PDF para alta qualidade
        plt.close()

# --- 3. BARRA EMPILHADA 100% HORIZONTAL (COMPOSIÇÃO GLOBAL) ---
def gerar_stacked_horizontal_cientifico(df):
    counts = df['final_sentiment'].value_counts(normalize=True) * 100
    ordem = ['POSITIVO', 'NEUTRO', 'NEGATIVO']
    vals = counts.reindex(ordem)

    fig, ax = plt.subplots(figsize=(8, 2))
    left = 0
    for s in ordem:
        ax.barh(0, vals[s], left=left, color=cores_cientificas[s], edgecolor='black', label=s)
        # Adiciona o texto apenas se houver espaço
        if vals[s] > 10:
            ax.text(left + vals[s]/2, 0, f'{vals[s]:.1f}%', ha='center', va='center', 
                    color='black' if s == 'NEGATIVO' else 'white', fontweight='bold')
        left += vals[s]

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel('Proportion (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False)
    plt.title('Global Sentiment Distribution')
    
    plt.savefig(os.path.join(PASTA_SAIDA, 'fig_global_distribution.pdf'), bbox_inches='tight')
    plt.close()

# --- 4. CONCORDÂNCIA DO ENSEMBLE (SIMPLES E DIRETO) ---
def gerar_concordancia_cientifica(df):
    auxiliares = ['llama', 'phi', 'deepseek']
    taxas = [round((df[mod] == df['mistral']).mean() * 100, 2) for mod in auxiliares]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([a.upper() for a in auxiliares], taxas, color='#666666', edgecolor='black', width=0.5)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_title('Agreement with Mistral (Primary Model)')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    for i, v in enumerate(taxas):
        ax.text(i, v + 2, f'{v}%', ha='center', fontsize=10)
    
    plt.savefig(os.path.join(PASTA_SAIDA, 'fig_agreement_ensemble.pdf'), bbox_inches='tight')
    plt.close()

# --- EXECUÇÃO ---
try:
    print("🎓 Gerando gráficos em padrão científico (PDF/White background)...")
    df_artigo = carregar_dados()
    gerar_barras_agrupadas(df_artigo)
    gerar_stacked_horizontal_cientifico(df_artigo)
    gerar_concordancia_cientifica(df_artigo)
    print(f"✅ Concluído! Gráficos prontos na pasta '{PASTA_SAIDA}'.")
except Exception as e:
    print(f"❌ Erro: {e}")