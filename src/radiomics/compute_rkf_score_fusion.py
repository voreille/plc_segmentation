from operator import mod
import os
import random
from itertools import product
from pathlib import Path

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

project_dir = Path(__file__).resolve().parents[2]
path_to_features = project_dir / "data/processed/radiomics/extracted_features.csv"
path_to_outcomes = project_dir / "data/clinical_info.csv"

model_type = "light"
n_splits = 10
n_repeats = 10
vois = ["GTV_L", "GTV_T", "GTV_N"]



def main():
    df = load_data(path_to_features, path_to_outcomes)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits,
                                  n_repeats=n_repeats,
                                  random_state=4)
    scores = list()
    for voi in vois:
        scores.append(
            compute_rkf(
                rkf,
                df,
                voi=voi,
                model_pt=model_dict[model_type](),
                model_ct=model_dict[model_type](),
                model_fusion=model_dict[model_type](),
            ))

    scores = pd.concat(scores, ignore_index=True)
    scores.to_csv(project_dir /
                  f"results/radiomics/rkf_results/{model_type}_fusion.csv")


def compute_rkf(rkf,
                df,
                voi="GTV_L",
                model_ct=None,
                model_pt=None,
                model_fusion=None):
    X_pt, y, feature_names_pt = get_formatted_data(df, modality="PT", voi=voi)
    X_ct, _, feature_names_ct = get_formatted_data(df, modality="CT", voi=voi)

    score = pd.DataFrame()
    for i, (idx_train, idx_test) in enumerate(rkf.split(X_pt, y=y)):

        X_train_pt, X_test_pt = X_pt[idx_train], X_pt[idx_test]
        X_train_ct, X_test_ct = X_ct[idx_train], X_ct[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        model_pt.fit(X_train_pt, y_train)
        model_ct.fit(X_train_ct, y_train)

        X_train_ct = model_ct.best_estimator_[:-1].transform(X_train_ct)
        X_test_ct = model_ct.best_estimator_[:-1].transform(X_test_ct)

        X_train_pt = model_pt.best_estimator_[:-1].transform(X_train_pt)
        X_test_pt = model_pt.best_estimator_[:-1].transform(X_test_pt)

        X_train = np.concatenate([X_train_ct, X_train_pt], axis=1)
        X_test = np.concatenate([X_test_ct, X_test_pt], axis=1)

        model_fusion.fit(X_train, y_train)

        y_pred = model_fusion.predict(X_test)
        y_score = model_fusion.predict_proba(X_test)[:, 1]
        score = append_score(score,
                             y_test,
                             y_pred,
                             y_score,
                             split=i,
                             n_train=len(idx_train),
                             n_test=len(idx_test),
                             pos_label=1)
    score["modality"] = "early_fusion"
    score["voi"] = voi
    return score


def compute_rkf_dummy(rkf, df, strategy="stratified"):

    X, y, feature_names = get_formatted_data(df, modality="CT", voi="GTV_L")
    model = DummyClassifier(strategy=strategy)

    score = pd.DataFrame()
    for i, (idx_train, idx_test) in enumerate(rkf.split(X, y=y)):

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        score = append_score(score,
                             y_test,
                             y_pred,
                             y_score,
                             split=i,
                             n_train=len(idx_train),
                             n_test=len(idx_test),
                             pos_label=1)
    score["strategy"] = strategy
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


model_dict = {
    "default": get_default_model,
    "light": get_light_model,
}


def append_score(
    df,
    y_true,
    y_pred,
    y_score,
    split=None,
    n_train=-1,
    n_test=-1,
    pos_label=1,
):
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
            'split': split,
            'n_train': n_train,
            'n_test': n_test,
        },
        ignore_index=True)


if __name__ == '__main__':
    main()
