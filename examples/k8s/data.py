import pandas as pd

df = pd.read_csv("examples/data/sampled_app_train.csv")
df['_tmp'] = df['SK_ID_CURR'].apply(lambda x: list(range(10)))
edf = df.explode('_tmp')
edf = edf.drop(columns=['_tmp'])
edf.to_csv("examples/data/100k_sampled_app_train.csv", index=False)
