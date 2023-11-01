
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

np.random.seed(42)

class LoadDataset:
    """
    obtains split dataset (train, val, test)
    """

    def __init__(self):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.prepare_dataset()
    
    def prepare_dataset(self):
        train_df_raw = pd.read_csv("data/train.csv")

        # check and drop NA values
        train_df = train_df_raw.dropna()

        # obtain X, y
        X = train_df[['question1', 'question2']]
        y = train_df['is_duplicate']

        # tokenise X, y
        # to do this 

        # split train_df into X&y train, val, test
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    
    def main(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test


"""
to see output
train_df_raw = pd.read_csv("data/train.csv")
train_df = train_df_raw.dropna()
X = train_df[['question1', 'question2']]
y = train_df['is_duplicate']
# print(X.head())
# print(y.head())
"""