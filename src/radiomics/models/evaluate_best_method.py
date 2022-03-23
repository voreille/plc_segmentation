import os
import random
from pprint import pprint

from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    roc_auc_score, confusion_matrix
from scipy.stats import t

from src.radiomics.utils import RemoveHighlyCorrelatedFeatures
from src.radiomics.data import load_data, get_formatted_data

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)

path_to_features = "/home/valentin/python_wkspce/plc_segmentation/data/processed/radiomics/extracted_features.csv"
path_to_outcomes = "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info_updated.csv"

shuffle = True
n_splits = 10
n_repeats = 10
rope_interval = [-0.05, 0.05]


def main():
    df = load_data(path_to_features, path_to_outcomes)
    X1, y1, feature_names1 = get_formatted_data(df, modality="CT", voi="GTV_L")
    X2, y2, feature_names2 = get_formatted_data(df, modality="PT", voi="GTV_N")

    # max_indice = np.where(
    #     np.array(feature_names2) == "original_firstorder_Maximu")[0]
    # X2 = X2[:, max_indice]

    if shuffle:
        idx = np.arange(X1.shape[0])
        np.random.shuffle(idx)
        X1 = X1[idx, :]
        y1 = y1[idx]

        X2 = X2[idx, :]
        y2 = y2[idx]

    rkf = RepeatedStratifiedKFold(n_splits=n_splits,
                                  n_repeats=n_repeats,
                                  random_state=4)

    scores1 = compute_rkf_scores(X1, y1, rkf=rkf)
    scores2 = compute_rkf_scores(X2,
                                 y2,
                                 model=DummyClassifier(strategy="stratified"),
                                 rkf=rkf)

    n = n_splits * n_repeats
    n_train = len(list(rkf.split(X1, y1))[0][0])
    n_test = len(list(rkf.split(X1, y1))[0][1])

    t_test_results = dict()
    stats1 = dict()
    stats2 = dict()
    for col in scores1.columns:
        s1 = scores1[col].values
        s2 = scores2[col].values
        t_stat, p_val = compute_corrected_ttest(s1 - s2, n - 1, n_train,
                                                n_test)

        t_post = t(n - 1,
                   loc=np.mean(s1 - s2),
                   scale=corrected_std(s1 - s2, n_train, n_test))

        t_test_results.update({
            col: {
                "proba M1 > M2":
                1 - t_post.cdf(0),
                "ROPE interval":
                t_post.cdf(rope_interval[1]) - t_post.cdf(rope_interval[0]),
                "p-value":
                p_val,
                "t-value":
                t_stat
            }
        })
        stats1.update({
            col: {
                "mean": np.mean(s1),
                "5th quantile": np.quantile(s1, 0.05),
                "95th quantile": np.quantile(s1, 0.95)
            }
        })
        stats2.update({
            col: {
                "mean": np.mean(s2),
                "5th quantile": np.quantile(s2, 0.05),
                "95th quantile": np.quantile(s2, 0.95)
            }
        })

    print("T-test results")
    pprint(t_test_results)
    print("Stats Model 1")
    pprint(stats1)
    print("Stats Model 2")
    pprint(stats2)
    print("voila")


def compute_rkf_scores(X, y, model=None, rkf=None, dummy=False):
    if model is None:
        # model = get_default_model()
        model = get_light_model()
    score = pd.DataFrame()
    for idx_train, idx_test in rkf.split(X, y=y):

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        score = append_score(score, y_test, y_pred, y_score, pos_label=1)

    return score


def get_simplest_model():
    return Pipeline(steps=[
        ('normalization', StandardScaler()),
        ('classifier',
         LogisticRegression(
             penalty='l2',
             C=100000,
             solver='liblinear',
             max_iter=1000,
         )),
    ])


def get_default_model():
    scaler = StandardScaler()

    clf_lr = LogisticRegression(penalty='l2',
                                C=100000,
                                solver='liblinear',
                                max_iter=1000)
    clf_lr_l1 = LogisticRegression(penalty='l1', solver='liblinear')
    clf_rf = RandomForestClassifier()

    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_selection', None),  # SelectKBest(f_classif)),
        ('classifier', None),
    ])

    F_OPTIONS = [1, 3, 5, 10, "all"]
    C_OPTIONS = [10**k for k in range(-4, 5, 1)]
    K_OPTIONS = [k for k in range(1, 11)]

    param_grid = [
        {
            'feature_selection': [SelectKBest(f_classif)],
            'feature_selection__k': F_OPTIONS,
            'classifier': [clf_rf],
            'classifier__n_estimators': [100, 150]
        },
        {
            'feature_selection': [SelectKBest(f_classif)],
            'feature_selection__k': F_OPTIONS,
            'classifier': [clf_lr],
        },
        {
            'feature_selection': [SelectKBest(f_classif)],
            'feature_selection__k': F_OPTIONS,
            'classifier': [clf_lr_l1],
            'classifier__C': C_OPTIONS,
        },
        {
            'feature_selection': [PCA()],
            'feature_selection__n_components': K_OPTIONS,
            'classifier': [clf_lr],
        },
        {
            'feature_selection': [PCA()],
            'feature_selection__n_components': K_OPTIONS,
            'classifier': [clf_lr_l1],
            'classifier__C': C_OPTIONS,
        },
        {
            'feature_selection': [PCA()],
            'feature_selection__n_components': K_OPTIONS,
            'classifier': [clf_rf],
        },
    ]
    search = GridSearchCV(pipe,
                          param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=23,
                          refit=True,
                          verbose=1,
                          scoring="roc_auc")
    return search


def get_light_model():
    scaler = StandardScaler()
    clf_lr = LogisticRegression(penalty='l2',
                                C=100000,
                                solver='liblinear',
                                max_iter=1000)

    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_selection', None),  # SelectKBest(f_classif)),
        ('classifier', None),
    ])

    F_OPTIONS = [3, 4, 5]

    param_grid = [
        {
            'feature_selection': [SelectKBest(f_classif)],
            'feature_selection__k': F_OPTIONS,
            'classifier': [clf_lr],
        },
    ]

    search = GridSearchCV(pipe,
                          param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=23,
                          refit=True,
                          verbose=1,
                          scoring="roc_auc")
    return search


def corrected_std(differences, n_train, n_test):
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val


def print_results(d):
    for key, item in d.items():
        print(f"{key}: {item[0]:0.2f} ({item[1]:0.2f} - {item[2]:0.2f})")


def append_score(df, y_true, y_pred, y_score, pos_label=1, neg_label=0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = recall_score(y_true, y_pred, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    return df.append(
        {
            'roc_auc': roc_auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'accuracy': accuracy,
            'precision': precision,
            'npv': npv,
        },
        ignore_index=True)


if __name__ == '__main__':
    main()
