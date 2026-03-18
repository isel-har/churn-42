from sklearn.model_selection import train_test_split
import pandas as pd


class DataBalancer:
    def __init__(self, target_column='TARGET', random_state=42):
        self.target_column = target_column
        self.random_state  = random_state

    def balance_training_set(self, df, ratio=1.0, save=True):

        minority = df[df[self.target_column] == 1]
        majority = df[df[self.target_column] == 0]

        # Calculate how many majority samples we need to hit the ratio
        # count_maj = count_min / ratio
        target_majority_count = int(len(minority) / ratio)
        
        if target_majority_count > len(majority):
            print(f"Warning: Requested ratio {ratio} requires more majority samples than available. Using all.")
            target_majority_count = len(majority)

        majority_sampled = majority.sample(n=target_majority_count, random_state=self.random_state)

        balanced_df = pd.concat([minority, majority_sampled]) \
            .sample(frac=1, random_state=self.random_state) \
            .reset_index(drop=True)

        if save:
            balanced_df.to_csv("data/balanced_train_data.csv", index=False)

        return balanced_df


class DataSplitter:

    def __init__(self, val_size=0.1, test_size=0.2):
        self.val_size = val_size
        self.test_size = test_size


    def split(self, path, save_splits=True):

        df = pd.read_csv(path)

        df_, df_test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['TARGET'],
            random_state=42
        )

        df_train, df_val = train_test_split(
            df_,
            test_size=self.val_size, 
            stratify=df_['TARGET'],
            random_state=42
        )
    
        if save_splits:
            df_test.to_csv('data/split_data_test.csv', index=False)
            df_val.to_csv('data/split_data_test.csv', index=False)

        del df_val, df_test

        return df_train

