import pandas as pd
import numpy as np

n_samples = 70000
params = {
    'learning_rate': np.random.lognormal(-5, 1, n_samples),
    'batch_size': np.random.choice([32, 64, 128, 256], n_samples),
    'num_layers': np.random.randint(2, 8, n_samples),
    'dropout_rate': np.random.beta(2, 5, n_samples),
    'activation': np.random.choice(['relu', 'selu', 'leaky_relu', 'gelu'], n_samples)
}

# Função objetivo complexa
params_df = pd.DataFrame(params)
params_df['acuracia'] = (
    0.8
    - 0.1 * np.log(params_df['learning_rate'])
    + 0.05 * params_df['num_layers']
    - 0.2 * params_df['dropout_rate']
    + np.where(params_df['activation'] == 'gelu', 0.3, 0)
    + np.random.normal(0, 0.02, n_samples)
)

# Salvar em CSV
params_df.to_csv("dataset_otimizacao_bayesiana_hiperparametros_de_deep_learning.csv", index=False)

# Salvar em parquet
params_df.to_parquet("dataset_otimizacao_bayesiana_hiperparametros_de_deep_learning.parquet", engine='pyarrow',
                     compression='zstd')