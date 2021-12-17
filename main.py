from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import aif360.sklearn.metrics as metrics


VERBOSE = True # verbose training
SEED = 123
NEIGHBORS = 5 # KNN
NUMBER_OF_ROWS = 110000 # total data rows loaded
TRAIN_DATA_PATH = Path("./data_train.csv")
TRAIN_TEST_SPLIT = 0.6 # 90% train
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
APPLYING_FAIRNESS = True
PROTECTED_ATTRIBUTES = ["smoking", "background_diseases_binary"]
# 0 = Unlinking
# 1 = Reweighing
FAIRNESS_TYPE = 1


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
    data["country"].replace(other_countries, 1.0, inplace=True)
    data["country"].replace(MAIN_COUNTRY, 0.0, inplace=True)

def binarize_age(data, PRIV_AGE, UNPRIV_AGE):
    data["age"].replace(PRIV_AGE, 0, inplace=True)
    data["age"].replace(UNPRIV_AGE, 1, inplace=True)

def clean_data(data) -> pd.DataFrame:
    """apply one hot encoding and replace NaN entries"""
    data = pd.get_dummies(data)
    data = data.fillna(0)
    data.replace(0.75, 1, inplace=True) # we found this weird entry after long debugging
    return data

def get_fairness_weights(data, protected: str, target = TARGET) -> pd.DataFrame:
    """apply pre-processing fairness by dropping the protected attribute"""
    # num smokers * num cheap
    denominator_right = len(data[(data[protected] == 1) & (data[target] == 0)])
    if denominator_right == 0: denominator_right = 1
    priv_pos = (len(data[protected] == 1) * len(data[target] == 0)) / (len(data) * denominator_right)

    denominator_right = len(data[(data[protected] == 1) & (data[target] == 1)])
    if denominator_right == 0: denominator_right = 1
    priv_neg = (len(data[protected] == 1) * len(data[target] == 1)) / (len(data) * denominator_right)

    denominator_right = len(data[(data[protected] == 0) & (data[target] == 0)])
    if denominator_right == 0: denominator_right = 1
    unpriv_pos = (len(data[protected] == 0) * len(data[target] == 0)) / (len(data) * denominator_right)
    
    denominator_right = len(data[(data[protected] == 0) & (data[target] == 1)])
    if denominator_right == 0: denominator_right = 1
    unpriv_neg = (len(data[protected] == 0) * len(data[target] == 1)) / (len(data) * denominator_right)

    weight_list = []

    for index, row in data.iterrows():
        if row[protected] == 1 and row[target] == 0: weight_list.append(priv_pos)
        if row[protected] == 1 and row[target] == 1: weight_list.append(priv_neg)
        if row[protected] == 0 and row[target] == 0: weight_list.append(unpriv_pos)
        if row[protected] == 0 and row[target] == 1: weight_list.append(unpriv_neg)

    data[protected + "_weights"] = weight_list

    return data

def apply_weights(data, prot_attributes):
    """each protected attribute is multiplied by its weights"""
    for attr in prot_attributes:
        data[attr] = data[attr] * data[attr + "_weights"]
    return data

def split_train_test(X, y, split = TRAIN_TEST_SPLIT, seed = SEED) -> tuple:
    """split into train and test set"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed, shuffle=False)
    return X_train, X_test, y_train, y_test

def fit_model(X_train, y_train, n = NEIGHBORS) -> object:
    model = KNeighborsClassifier(n_neighbors=n)
    #model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def test_model(X_test, model, y_test) -> np.ndarray:
    labels = model.predict(X_test)
    # probs = model.predict_proba(X_test)
    # accuracy = model.score(X_test, y_test)
    return labels

def diff(a, b):
    return max(b-a, a-b)

def div(a, b):
    return min(a/b, b/a)

def metrics(X, y, t = TARGET, pos_label = 0):
    # the problem is that the X object starts at a different index than the y object in the test set
    X_reindex = X.reset_index(drop=True)
    y_reindex = y.reset_index(drop=True)
    data = pd.concat([X_reindex, y_reindex], axis=1)
    def gender(func):
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
        base1 = len(data.loc[data['background_diseases_binary'] == 0])
        group2 = data.loc[data["background_diseases_binary"] == 1][t].reindex(data.background_diseases_binary.unique(), fill_value=0)[pos_label]
        base2 = len(data.loc[data['background_diseases_binary'] == 1])
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
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # fairness
    if APPLYING_FAIRNESS:
        print("APPLYING FAIRNESS")
        if FAIRNESS_TYPE == 0:
            print("UNIFORM_RANDOMIZATION")
            rand_list = np.random.randint(2, size=len(X_train))
            for attr in PROTECTED_ATTRIBUTES:
                X_train[attr] = rand_list
        else:
            print("REWEIGHING")
            X_train = pd.concat([X_train, y_train], axis=1)
            for attr in PROTECTED_ATTRIBUTES:
                X_train = get_fairness_weights(X_train, attr, TARGET)
            X_train = X_train.drop([TARGET], axis=1)
            X_train = apply_weights(X_train, PROTECTED_ATTRIBUTES)
            for attr in PROTECTED_ATTRIBUTES:
                X_train = X_train.drop([attr+"_weights"], axis=1)

    print("TRAIN DATA")
    print(X_train)
    print("TEST DATA")
    print(X_test)

    # train and test
    model = fit_model(X_train, y_train)
    labels = test_model(X_test, model, y_test)
    labels = pd.DataFrame(labels, columns=["severity_illness"])
    print(classification_report(y_true=y_test, y_pred=labels, zero_division=0, target_names=tn))

    # fairness metrics
    if not APPLYING_FAIRNESS:
        print("####################################### TRAIN SET DISCRIMININATION")
        (mean_diff_gender, mean_diff_smoking, mean_diff_background, mean_diff_age, mean_diff_country,
        imp_ratio_gender, imp_ratio_smoking, imp_ratio_background, imp_ratio_age, imp_ratio_country,) = metrics(X_train, y_train)
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
        print("")

    print("####################################### TEST SET DISCRIMININATION")
    (mean_diff_gender, mean_diff_smoking, mean_diff_background, mean_diff_age, mean_diff_country,
    imp_ratio_gender, imp_ratio_smoking, imp_ratio_background, imp_ratio_age, imp_ratio_country,) = metrics(X_test, labels)
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
    print("")

main()
