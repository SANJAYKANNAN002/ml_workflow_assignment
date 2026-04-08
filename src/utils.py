import pandas as pd


def check_missing_values(df):
    """
    Returns missing values in each column
    """
    return df.isnull().sum()


def get_basic_info(df):
    """
    Prints basic dataset information
    """
    print("\nDataset Info:")
    print(df.info())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nSummary Statistics:")
    print(df.describe())


def check_class_distribution(y):
    """
    Prints distribution of target variable
    """
    print("\nTarget Variable Distribution:")
    print(y.value_counts())