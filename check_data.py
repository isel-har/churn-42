
import pandas as pd


import sys

df = pd.read_csv(sys.argv[1], usecols=['TARGET'])


print("train dataset target count")
print(len(df[df['TARGET'] == 0]))
print(len(df[df['TARGET'] == 1]))


df = pd.read_csv(sys.argv[2], usecols=['TARGET'])


print("test dataset target count")
print(len(df[df['TARGET'] == 0]))
print(len(df[df['TARGET'] == 1]))