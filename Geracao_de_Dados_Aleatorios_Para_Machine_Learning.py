import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import gc  # Garbage Collector

# Configurar seed para reprodutibilidade
np.random.seed(42)


def generate_data_in_chunks(total_samples, chunk_size=10000):
    """
    Gera dados em chunks para melhor gerenciamento de memória
    """
    chunks = []
    num_chunks = total_samples // chunk_size

    print("Gerando dados em chunks...")
    for i in tqdm(range(num_chunks)):
        # Gerar timestamps para o chunk atual
        start_date = datetime(2023, 1, 1) + timedelta(days=i * chunk_size)
        dates = [start_date + timedelta(days=x) for x in range(chunk_size)]

        # Dados do chunk
        chunk_data = {
            'timestamp': dates,
            'user_id': np.random.randint(10000, 99999, chunk_size),
            'categoria': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], chunk_size,
                                          p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),  # Distribuição mais realista
            'valor_compra': np.clip(np.random.normal(150, 50, chunk_size), 10, 500),  # Valores entre 10 e 500
            'idade_cliente': np.random.normal(35, 12, chunk_size).astype(int).clip(18, 90),
            # Distribuição normal para idades
            'score_credito': np.random.normal(700, 50, chunk_size).clip(300, 850),  # Score de crédito mais realista
            'frequencia_compras': np.random.negative_binomial(5, 0.5, chunk_size),
            # Distribuição mais realista para frequência
            'satisfacao': np.random.choice([1, 2, 3, 4, 5], chunk_size,
                                           p=[0.05, 0.1, 0.2, 0.4, 0.25]),
            # Distribuição enviesada para satisfação alta
            'tempo_cliente_dias': np.random.exponential(500, chunk_size).astype(int).clip(1, 3650),  # Máximo 10 anos
            'probabilidade_retorno': np.random.beta(8, 2, chunk_size),  # Distribuição beta para probabilidade
            'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste', 'Centro'], chunk_size),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], chunk_size, p=[0.6, 0.3, 0.1]),
            'metodo_pagamento': np.random.choice(['Crédito', 'Débito', 'Pix', 'Boleto'], chunk_size,
                                                 p=[0.4, 0.3, 0.2, 0.1])
        }

        # Criar DataFrame do chunk
        chunk_df = pd.DataFrame(chunk_data)

        # Adicionar features derivadas
        chunk_df['valor_total'] = chunk_df['valor_compra'] * chunk_df['frequencia_compras']
        chunk_df['score_final'] = (chunk_df['score_credito'] / 850 + chunk_df['satisfacao'] / 5) / 2
        chunk_df['is_vip'] = (chunk_df['valor_total'] > chunk_df['valor_total'].quantile(0.8)).astype(int)
        chunk_df['lifetime_value'] = chunk_df['valor_total'] * (chunk_df['tempo_cliente_dias'] / 365)
        chunk_df['engagement_score'] = (chunk_df['frequencia_compras'] * chunk_df['satisfacao'] *
                                        chunk_df['probabilidade_retorno']).round(2)

        # Arredondar valores numéricos
        chunk_df['valor_compra'] = chunk_df['valor_compra'].round(2)
        chunk_df['score_credito'] = chunk_df['score_credito'].round(0)
        chunk_df['probabilidade_retorno'] = chunk_df['probabilidade_retorno'].round(3)
        chunk_df['valor_total'] = chunk_df['valor_total'].round(2)
        chunk_df['score_final'] = chunk_df['score_final'].round(3)
        chunk_df['lifetime_value'] = chunk_df['lifetime_value'].round(2)

        # Adicionar valores ausentes de forma estratificada
        for col in chunk_df.columns:
            if col not in ['timestamp', 'user_id']:  # Preservar colunas críticas
                mask = np.random.random(size=chunk_size) < 0.03  # 3% de valores ausentes
                chunk_df.loc[mask, col] = np.nan

        chunks.append(chunk_df)

        # Limpar memória
        gc.collect()

    # Combinar todos os chunks
    print("Combinando chunks...")
    final_df = pd.concat(chunks, ignore_index=True)

    # Adicionar alguns outliers controlados
    print("Adicionando outliers controlados...")
    outliers_idx = np.random.choice(final_df.index, size=int(total_samples * 0.01), replace=False)
    final_df.loc[outliers_idx, 'valor_compra'] *= np.random.uniform(5, 10, size=len(outliers_idx))
    final_df.loc[outliers_idx, 'frequencia_compras'] *= np.random.uniform(3, 5, size=len(outliers_idx))

    return final_df


# Gerar dados
n_samples = 70000
chunk_size = 10000

print(f"Iniciando geração de {n_samples} amostras...")
df = generate_data_in_chunks(n_samples, chunk_size)

# Salvar em diferentes formatos
print("Salvando dados...")
df.to_csv('dados_clientes_large.csv', index=False)
df.to_parquet('dados_clientes_large.parquet')  # Formato mais eficiente para grandes volumes

# Mostrar informações sobre os dados
print("\nInformações do DataFrame:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nDistribuição de categorias:")
print(df['categoria'].value_counts(normalize=True))

print("\nDistribuição de métodos de pagamento:")
print(df['metodo_pagamento'].value_counts(normalize=True))

print("\nTamanho do arquivo CSV:", round(df.memory_usage().sum() / 1024 ** 2, 2), "MB")

print("\nArquivos 'dados_clientes_large.csv' e 'dados_clientes_large.parquet' foram gerados com sucesso!")