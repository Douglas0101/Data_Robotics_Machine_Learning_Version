import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Parâmetros
n_samples = 70000
np.random.seed(42)

# Gerar features sintéticas
X, _ = make_classification(n_samples=n_samples, n_features=15, n_informative=8, n_redundant=3, random_state=42)

# Engenharia de features realista
df = pd.DataFrame(X, columns=[
    "tamanho_m2", "idade_imovel", "distancia_centro", "num_quartos", 
    "num_banheiros", "garagem", "seguranca_24h", "area_lazer",
    "metro_proximo", "escola_proxima", "poluicao_sonora", "indice_criminal",
    "taxa_juros", "renda_media_bairro", "crescimento_populacional"
])

# Garantir que "taxa_juros" seja positiva
df["taxa_juros"] = np.abs(df["taxa_juros"])  # Taxa de juros não pode ser negativa

# Target não-linear com ruído heteroscedástico
df["preco"] = (
    1e5 * df["tamanho_m2"] 
    - 5e3 * df["idade_imovel"]**1.5 
    + 2e4 * (df["seguranca_24h"] * df["area_lazer"]) 
    + np.exp(0.1 * df["renda_media_bairro"]) 
    + np.random.normal(0, np.abs(5e4 + 1e4 * df["taxa_juros"]), n_samples)  # Garantir scale >= 0
)

# Salvar em Parquet
df.to_parquet("dataset_regressao_bayesiana_mercado_imobiliario.parquet", engine='pyarrow', compression='zstd')