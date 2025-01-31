import pandas as pd
import numpy as np
from faker import Faker


fake = Faker('pt_BR')
np.random.seed(42)

# Gerar dados
n_samples = 70000
data = []

for _ in range(n_samples):
    categoria = np.random.choice(["spam", "notícia", "review positivo", "review negativo"], p=[0.3, 0.2, 0.3, 0.2])

    if categoria == "spam":
        texto = fake.bs() + " " + fake.company_email()
    elif categoria == "notícia":
        texto = fake.sentence(nb_words=12) + " " + fake.date(pattern="%d/%m/%Y")
    else:
        texto = fake.text(max_nb_chars=200) if categoria == "review positivo" else fake.text(max_nb_chars=200
                                                                                             )[:50] + " péssimo "
    data.append({
        "texto": texto,
        "categoria": categoria,
        "data_envio": fake.date_time_this_year(),
        "user_id": fake.uuid4(),
        "prioridade": np.random.poisson(lam=2) + 1
    })

# Salvar em CSV
df = pd.DataFrame(data)
df.to_csv("dataset_naive_bayes_text_classification.csv")

# Salvar em Parquet (com compressão Snappy)
df = pd.DataFrame(data)
df.to_parquet("dataset_naive_bayes_text_classification.parquet", engine="pyarrow", compression="zstd")