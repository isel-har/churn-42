
import pandas as pd
import sys
import gc
from preprocessors import MissingAwareColumnSelector
import joblib

pd.set_option('display.max_rows', None)

def main():

    if len(sys.argv) != 2:
        sys.exit(1)
    
    try:
        
        chunksize   = 50_000
        majority_sample_fraction = 0.26
        missing_threshold = 0.7


        sampled_chunks = []

        for chunk in pd.read_csv(sys.argv[1], chunksize=chunksize):
    
            minority = chunk[chunk['TARGET'] == 1]
            majority = chunk[chunk['TARGET'] == 0]
            
            majority_sampled = majority.sample(frac=majority_sample_fraction, random_state=42)

    
            combined_chunk = pd.concat([minority, majority_sampled])

            sampled_chunks.append(combined_chunk)

            del chunk, minority, majority, combined_chunk
            gc.collect()


        df_balanced = pd.concat(sampled_chunks, axis=0)

        missing_ratio = df_balanced.isnull().mean()

        cols_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        cols_drop.append('ID')
        # cols_drop.append('ID')
        df_balanced = df_balanced.drop(columns=cols_drop)
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # IMPORTANT: Shuffle the dataframe 
        # # (Since we processed in chunks, the classes might be ordered)


        selector = MissingAwareColumnSelector(y_cols=['TARGET'], missing_threshold=missing_threshold)
        selector.fit(df_balanced)
    
    
        joblib.dump(selector, 'selector.pkl')
        df_balanced.to_csv("data/balanced_bank_data_train.csv", index=False)
        
    except Exception as e:
        print("exception :", str(e))


if __name__ == "__main__":
    main()

