from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import aif360.sklearn.metrics as metrics

VERBOSE = True # verbose training
SEED = 123
NEIGHBORS = 3 # KNN
NUMBER_OF_ROWS = 20000 # total data rows loaded
TRAIN_DATA_PATH = Path("./data_train.csv")
TRAIN_TEST_SPLIT = 0.9 # 90% train
COLUMNS = ["age", "country", 
    "sex", "smoking",
    "background_diseases_diabetes",
    "background_diseases_hypertension",
    "background_diseases_kidney_failure",
    "background_diseases_asthma",
    "background_diseases_cardiovascular",
    "background_diseases_obesity",
    "background_diseases_immunosuppression",
    "background_diseases_chronic_obstructive_pulmonary",
    "severity_illness"]
TARGET = "severity_illness" # target variable
TARGET_NAMES = ["asymptomatic", "critical", "cured", "deceased", "good"] # oder matters

def import_data(train_path = TRAIN_DATA_PATH, nrows = NUMBER_OF_ROWS) -> pd.DataFrame:
    """load CSVs into dataframes"""
    df = pd.read_csv(train_path, nrows=nrows)
    return df

def filter_data(data, keep_cols = COLUMNS) -> pd.DataFrame:
    """remove unwanted columns"""
    data = data[keep_cols]
    return data

def separate_X_y(data, target=TARGET) -> pd.DataFrame:
    """separete X (predictors) from y (label)"""
    y = data[target]
    X = data.drop([target], axis=1)
    return X, y

def clean_data(data) -> pd.DataFrame:
    """apply one hot encoding and replace NaN entries"""
    data = pd.get_dummies(data)
    data = data.fillna(0)
    return data

def split_train_test(X, y, split = TRAIN_TEST_SPLIT, seed = SEED) -> tuple:
    """split into train and test set"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)
    return X_train, X_test, y_train, y_test

def fit_model(X_train, y_train, n = NEIGHBORS) -> object:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    return model

def test_model(X_test, model) -> np.ndarray:
    labels = model.predict(X_test)
    # probs = model.predict_proba(X_test)
    # accuracy = model.score(X_test, y_test)
    return labels

def binarize_data(data) -> np.nadarray:
    NotImplemented


def main(tn = TARGET_NAMES):
    data = import_data()
    data = filter_data(data)
    # first split X and y, then preprocess, then split train and test!
    X, y = separate_X_y(data)
    X, y = map(clean_data, [X, y])
    X_train, X_test, y_train, y_test = split_train_test(X, y) 
    model = fit_model(X_train, y_train)
    labels = test_model(X_test, model)
    
    print(classification_report(y_true=y_test, y_pred=labels, zero_division=0, target_names=tn))


main()
