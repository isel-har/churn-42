from data_preparation import DataSplitter, DataBalancer
import pandas as pd
import sys


pd.set_option('display.max_rows', None)


def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    try:

        splitter = DataSplitter()

        train_df = splitter.split(path=sys.argv[1])
      
        balancer = DataBalancer()

        df_train_balanced = balancer.balance_training_set(df=train_df, ratio=0.5, save=True)

        print("New Distribution:")
        print(df_train_balanced['TARGET'].value_counts(normalize=True))

    except Exception as e:
        print("exception :", str(e))

if __name__ == "__main__":
    main()
