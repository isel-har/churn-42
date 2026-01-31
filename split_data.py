import pandas as pd

samples = 0 
for chunk in pd.read_csv("data/bank_data_train.csv", chunksize=100_000):
    samples += len(chunk)


test_size = (samples * 20) / 100

df = pd.read_csv("data/bank_data_train.csv", nrows=test_size)

df.to_csv("data/test_data.csv")