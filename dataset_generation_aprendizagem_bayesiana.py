import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

class BayesianDatasetGenerator:
    def __init__(self, n_samples=5000, noise_level=0.2, seed=42):
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.seed = seed
        np.random.seed(seed)

    def generate_complex_features(self):
        # Dados binários com correlação não linear
        X_bin, y_bin = make_classification(
            n_samples=self.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_classes=2,
            flip_y=self.noise_level,
            random_state=self.seed
        )

        # Dados contínuos para regressão hierárquica
        X_reg, y_reg = make_regression(
            n_samples=self.n_samples,
            n_features=3,
            noise=self.noise_level * 10,
            random_state=self.seed
        )

        # Grupos hierárquicos (3 clusters)
        groups = np.random.choice(['A', 'B', 'C'], size=self.n_samples, p=[0.4, 0.3, 0.3])
        group_effect = np.where(groups == 'A', 1.5, np.where(groups == 'B', -0.5, 0.0))
        y_reg += group_effect + np.random.normal(0, 2, self.n_samples)

        # Time-series sintética
        time = np.arange(self.n_samples)
        trend = 0.1 * time
        seasonality = 10 * np.sin(2 * np.pi * time / 365)
        y_time_series = trend + seasonality + np.random.normal(0, 5, self.n_samples)

        # DataFrame final
        df = pd.DataFrame({
            'feature_bin1': X_bin[:, 0],
            'feature_bin2': X_bin[:, 1],
            'target_bin': y_bin,
            'feature_reg1': X_reg[:, 0],
            'feature_reg2': X_reg[:, 1],
            'feature_reg3': X_reg[:, 2],
            'target_reg': y_reg,
            'group': groups,
            'time': time,
            'target_time_series': y_time_series
        })

        # Adicionar missing values (5% aleatório)
        df = df.mask(np.random.random(df.shape) < 0.05)

        return df

    def save_dataset(self, df, path='data/synthetic_dataset.csv'):
        df.to_csv(path, index=False)
        print(f'Dataset salvo em {path}')

    def save_dataset_parquet(self, df, path='data/synthetic_dataset.parquet'):
        df.to_parquet(path, engine='pyarrow', index=False)
        print(f'Dataset salvo em {path}')

# Teste de funcionamento
if __name__ == '__main__':
    generator = BayesianDatasetGenerator()
    df = generator.generate_complex_features()
    generator.save_dataset(df)
    print('Dataset gerado com colunas:', df.columns.tolist())