import pandas as pd

df = pd.read_csv(
    "hf://datasets/einrafh/hnm-fashion-recommendations-data/data/raw/articles.csv"
)

print(df.columns.tolist())


print(df.head(5))
