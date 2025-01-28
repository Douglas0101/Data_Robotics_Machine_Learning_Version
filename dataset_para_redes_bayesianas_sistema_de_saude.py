from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
import pandas as pd
import numpy as np

# Carregar modelo e gerar amostras
model = get_example_model('asia')
sampler = BayesianModelSampling(model)
samples_df = sampler.forward_sample(size=70000)

# Converter para DataFrame
samples = pd.DataFrame(samples_df)


# Garantir tipos categóricos
for col in samples.columns:
    samples[col] = samples[col].astype('category')

# Adicionar missing values (5%) de forma estruturada
for col in samples.columns:
    mask = np.random.rand(len(samples)) < 0.95
    samples.loc[~mask, col] = np.nan

# Forçar consistência lógica em 'either'
samples['either'] = samples['tub'].cat.codes | samples['lung'].cat.codes
samples['either'] = samples['either'].astype('category')

# Salvar em CSV
samples.to_csv("dataset_redes_bayesianas_sistema_de_saude.csv", index=False)

# Salvar em parquet
samples.to_parquet("dataset_redes_bayesianas_sistema_de_saude.parquet", engine='pyarrow', compression='zstd')