import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def preprocess_data(df):
    columns_to_remove = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    categorical_columns = ['Sex', 'Embarked']

    for c in columns_to_remove:
        df = df.drop(c, 1)

    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)

    fare_median = df['Fare'].median()
    df['Fare'].fillna(fare_median, inplace=True)

    return pd.get_dummies(df, dummy_na=True, columns=categorical_columns)


def load_data():
    return pd.read_csv('data/train.csv')


def get_train(df):
    train = preprocess_data(df)
    train = train.drop('Survived', 1)
    return train


def get_target(df):
    return df['Survived']


def get_model():
    return RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
