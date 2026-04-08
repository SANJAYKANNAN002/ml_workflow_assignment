import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Drop data leakage column
    df = df.drop(columns=["discount_used_on_repeat_order"])

    # Features and target
    X = df.drop(columns=["repeat_purchase_flag", "customer_id"])
    y = df["repeat_purchase_flag"]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)