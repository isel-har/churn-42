
from sklearn.model_selection import train_test_split
from preprocessors import MissingAwareColumnSelector
import pandas as pd
import joblib
import sys
import gc

pd.set_option('display.max_rows', None)

def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    try:
        majority_sample_fraction = 0.25

        df = pd.read_csv(sys.argv[1])

        df_train, df_test = train_test_split(
            df,
            test_size=0.2,
            stratify=df['TARGET'],
            random_state=42
        )

        minority = df_train[df_train['TARGET'] == 1]
        majority = df_train[df_train['TARGET'] == 0]
        majority_sampled = majority.sample(frac=majority_sample_fraction, random_state=42)

        df_balanced = pd.concat([minority, majority_sampled]) \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

        print(df_balanced.shape)
        print(df_balanced.describe())

        df_balanced.to_csv("data/balanced_bank_data_train.csv", index=False)
        df_test.to_csv("data/balanced_bank_data_test.csv", index=False)

        print("train/test split saved.")

    except Exception as e:
        print("exception :", str(e))

if __name__ == "__main__":
    main()
