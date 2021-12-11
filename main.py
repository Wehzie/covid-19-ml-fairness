from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import aif360.sklearn.metrics as metrics
from sklearn.utils.sparsefuncs import inplace_column_scale

VERBOSE = True # verbose training
SEED = 123
NEIGHBORS = 3 # KNN
NUMBER_OF_ROWS = 110000 # total data rows loaded
TRAIN_DATA_PATH = Path("./data_train.csv")
TRAIN_TEST_SPLIT = 0.9 # 90% train
COLUMNS = ["age", "country", 
    "sex", "smoking", "background_diseases_binary",
    "severity_illness"]
MAIN_COUNTRY = "colombia" # country with most data. colombia = 0, other = 1
TARGET = "severity_illness" # target variable
TARGET_NAMES_MULTI = ["asymptomatic", "critical", "cured", "deceased", "good"] # oder matters
TARGET_POS = ["asymptomatic", "cured"]
TARGET_NEG = ["critical", "deceased", "good"]
TARGET_NAMES = ["0", "1"] # 0: cheap, 1: expensive
PRIV_GENDER = "male" # 0
UNPRIV_GENDER = "female" # 1
PRIV_AGE = range(70) # 0
UNPRIV_AGE = range(70, 200) # 1


def import_data(train_path = TRAIN_DATA_PATH, nrows = NUMBER_OF_ROWS) -> pd.DataFrame:
    """load CSVs into dataframes"""
    df = pd.read_csv(train_path, nrows=nrows)
    return df

def get_other_countries(data, MAIN_COUNTRY):
    all_countries = set(data["country"])
    all_countries.remove(MAIN_COUNTRY)
    return all_countries

def remove_transgender(data) -> None:
    data.drop(data[data.sex == "transgender"].index, inplace=True)

def binarize_gender(data, PRIV_GENDER, UNPRIV_GENDER):
    data.replace(PRIV_GENDER, 0, inplace=True) # men: 0
    data.replace(UNPRIV_GENDER, 1, inplace=True) # women: 1

def filter_data(data, keep_cols = COLUMNS) -> pd.DataFrame:
    """remove unwanted columns"""
    data = data[keep_cols]
    return data

def separate_X_y(data, target=TARGET) -> pd.DataFrame:
    """separete X (predictors) from y (label)"""
    y = data[target]
    X = data.drop([target], axis=1)
    return X, y

def binarize_target(data, pos = TARGET_POS, neg = TARGET_NEG) -> np.ndarray:
    """group cheap and expensive patients into two groups"""
    data = data.replace(pos, 0) # cheap: asymptomatic, cured
    data = data.replace(neg, 1) # expensive: deceased, good, critical
    return data

def binarize_country(data, MAIN_COUNTRY, other_countries):
    data["country"].replace(other_countries, 1, inplace=True)
    data["country"].replace(MAIN_COUNTRY, 0, inplace=True)

def binarize_age(data, PRIV_AGE, UNPRIV_AGE):
    data["age"].replace(PRIV_AGE, 0, inplace=True)
    data["age"].replace(UNPRIV_AGE, 1, inplace=True)

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
    #model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def test_model(X_test, model) -> np.ndarray:
    labels = model.predict(X_test)
    # probs = model.predict_proba(X_test)
    # accuracy = model.score(X_test, y_test)
    return labels

def diff(a, b):
    return b-a

def div(a, b):
    return a/b

def metrics(X, y, t = TARGET, pos_label = 0):
    data = pd.concat([X, y], axis=1)
    def gender(func):
        #unpriv = data.loc[data['sex'] == 1]
        #print(sum(unpriv[t] == pos_label))
        group1 = data.loc[data['sex'] == 1][t].value_counts()[pos_label] # cheap|woman
        base1 = len(data.loc[data['sex'] == 1])
        group2 = data.loc[data['sex'] == 0][t].value_counts()[pos_label] # cheap|man
        base2 = len(data.loc[data['sex'] == 0])
        return func((group1/base1), (group2/base2))
    def smoking(func):
        group1 = data.loc[data["smoking"] == 0][t].value_counts()[pos_label]
        base1 = len(data.loc[data["smoking"] == 0])
        # infinity means approaches zero -> very unfair
        group2 = data.loc[data["smoking"] == 1][t].value_counts().reindex(data.smoking.unique(), fill_value=0)[pos_label]
        base2 = len(data.loc[data["smoking"] == 1])
        return func((group2/base2), (group1/base1))
    def background(func):
        group1 = data.loc[data["background_diseases_binary"] == 0][t].value_counts()[pos_label]
        print("healthy cheap", group1)
        base1 = len(data.loc[data['background_diseases_binary'] == 0])
        print("healthy", base1)
        group2 = data.loc[data["background_diseases_binary"] == 1][t].value_counts()[pos_label]
        print("ill cheap", group2)
        base2 = len(data.loc[data['background_diseases_binary'] == 1])
        print("ill", base2)
        return func((group2/base2), (group1/base1))
    def age(func):
        group1 = data.loc[data["age"] == 0][t].value_counts()[pos_label]
        base1 = len(data.loc[data["age"] == 0])
        group2 = data.loc[data["age"] == 1][t].value_counts()[pos_label]
        base2 = len(data.loc[data["age"] == 1])
        return func((group2/base2), (group1/base1))
    def country(func):
        group1 = data.loc[data["country"] == 0][t].value_counts()[pos_label]
        base1 = len(data.loc[data["country"] == 0])
        group2 = data.loc[data["country"] == 1][t].value_counts()[pos_label]
        base2 = len(data.loc[data["country"] == 1])
        return func((group2/base2), (group1/base1))
        
    return (gender(diff), smoking(diff), background(diff), age(diff), country(diff),
            gender(div), smoking(div), background(div), age(div), country(div))



def main(tn = TARGET_NAMES):
    data = import_data()
    other_countries = get_other_countries(data, MAIN_COUNTRY)
    remove_transgender(data)
    binarize_gender(data, PRIV_GENDER, UNPRIV_GENDER)
    data = filter_data(data)
    # first split X and y, then preprocess, then split train and test!
    X, y = separate_X_y(data)
    y = binarize_target(y)
    binarize_country(X, MAIN_COUNTRY, other_countries)
    binarize_age(X, PRIV_AGE, UNPRIV_AGE)
    X = clean_data(X)
    print(X.columns)
    X_train, X_test, y_train, y_test = split_train_test(X, y) 
    model = fit_model(X_train, y_train)
    labels = test_model(X_test, model)

    print(X_train)

    print(classification_report(y_true=y_test, y_pred=labels, zero_division=0, target_names=tn))

    # METRICS
    (mean_diff_gender, mean_diff_smoking, mean_diff_background, mean_diff_age, mean_diff_country,
    imp_ratio_gender, imp_ratio_smoking, imp_ratio_background, imp_ratio_age, imp_ratio_country,) = metrics(X, y)
    print("mean_diff_gender:", mean_diff_gender)
    print("mean_diff_smoking:", mean_diff_smoking)
    print("mean_diff_background:", mean_diff_background)
    print("mean_diff_age:", mean_diff_age)
    print("mean_diff_country:", mean_diff_country)

    print("imp_ratio_gender:", imp_ratio_gender)
    print("imp_ratio_smoking:", imp_ratio_smoking)
    print("imp_ratio_background:", imp_ratio_background)
    print("imp_ratio_age:", imp_ratio_age)
    print("imp_ratio_country:", imp_ratio_country)

    #print("base_rate:", metrics.base_rate(y_true=y_test, y_pred=labels, pos_label=0))
    #print("false_negative_error:", metrics.false_negative_rate_error(y_true=y_test, y_pred=labels, pos_label=0))
    #print("false_positive_rate_error", metrics.false_positive_rate_error(y_true=y_test, y_pred=labels, pos_label=0))

main()
