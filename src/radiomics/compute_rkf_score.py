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
from neuroCombat import neuroCombat

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
modalities = ["CT", "PT"]
# vois = ["GTV_L"]
vois = ["GTV_L", "GTV_T", "GTV_N"]

store_dummy = False
store_standard = False
store_suvmax = False
store_combat = True


def main():
    df = load_data(path_to_features,
                   path_to_outcomes,
                   clinical_info=["plc_status", "is_chuv"])
    rkf = RepeatedStratifiedKFold(n_splits=n_splits,
                                  n_repeats=n_repeats,
                                  random_state=4)
    scores = list()
    if store_standard:
        for modality, voi in product(modalities, vois):
            scores.append(
                compute_rkf(
                    rkf,
                    df,
                    modality=modality,
                    voi=voi,
                    model=model_dict[model_type](),
                ))

        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(project_dir /
                      f"results/radiomics/rkf_results/{model_type}.csv")
    if store_combat:
        df_scanner = pd.read_csv(
            "/home/valentin/python_wkspce/plc_segmentation/data/scanner_labels.csv"
        ).set_index("patient_id")
        for modality, voi in product(modalities, vois):
            scores.append(
                compute_rkf(rkf,
                            df,
                            modality=modality,
                            voi=voi,
                            model=model_dict[model_type](),
                            df_scanner=df_scanner))

        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(project_dir /
                      f"results/radiomics/rkf_results/{model_type}_combat.csv")

    if store_dummy:
        scores = pd.concat(
            [
                compute_rkf_dummy(rkf, df, strategy="stratified"),
                compute_rkf_dummy(rkf, df, strategy="prior"),
                compute_rkf_dummy(rkf, df, strategy="random_features"),
                compute_rkf_center(rkf, df),
            ],
            ignore_index=True,
        )

        scores.to_csv(project_dir / "results/radiomics/rkf_results/dummy.csv")
    if store_suvmax:
        scores = pd.concat(
            [
                compute_rkf_suvmax(rkf, df, strategy="bare"),
                compute_rkf_suvmax(rkf, df, strategy="ratio"),
                compute_rkf_suvmax(rkf, df, strategy="bare_gtvn"),
            ],
            ignore_index=True,
        )

        scores.to_csv(project_dir / "results/radiomics/rkf_results/suvmax.csv")


def compute_rkf_suvmax(rkf, df, strategy="bare"):
    X_gtvl, y, feature_names_gtvl = get_formatted_data(df,
                                                       modality="PT",
                                                       voi="GTV_L")

    X_gtvn, _, feature_name_gtvn = get_formatted_data(df,
                                                      modality="PT",
                                                      voi="GTV_N")
    max_indice = np.where(
        np.array(feature_names_gtvl) == "original_firstorder_Maximum")[0]
    if strategy == "ratio":
        X = X_gtvl[:, max_indice] / X_gtvn[:, max_indice]
    elif strategy == "bare":
        X = X_gtvl[:, max_indice]
    elif strategy == "bare_gtvn":
        X = X_gtvn[:, max_indice]

    model = get_simplest_model()

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


def compute_rkf(rkf,
                df,
                modality="CT",
                voi="GTV_L",
                model=None,
                df_scanner=None):
    X, y = get_formatted_data(df, modality=modality, voi=voi, return_df=True)
    if df_scanner is not None:
        X, y = apply_combat(X, y, df_scanner)
    else:
        X, y = X.values, y.values

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
    score["modality"] = modality
    score["voi"] = voi
    return score


def compute_rkf_center(rkf, df):
    df = df[(df["modality"] == "CT") & (df["voi"] == "GTV_L")]
    df = df.set_index("patient_id").sort_index()

    patient_list = [
        p for p in df.index
        if p not in ["PatientLC_63", "PatientLC_21", "PatientLC_71"]
    ]
    outcomes_df = df["plc_status"]
    df = df.drop(["plc_status", "voi", "modality"], axis=1)

    df = df.loc[patient_list, :]
    outcomes_df = outcomes_df[patient_list]

    X = df["is_chuv"].values
    X = X[..., np.newaxis]
    y = outcomes_df.values
    model = get_simplest_model()

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
    score["strategy"] = "center"
    return score


def compute_rkf_dummy(rkf, df, strategy="stratified"):

    X, y, feature_names = get_formatted_data(df, modality="CT", voi="GTV_L")
    if strategy != "random_features":
        model = DummyClassifier(strategy=strategy)
    else:
        model = get_light_model()
        X = np.random.uniform(size=X.shape)

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


def apply_combat(X, y, df_scanner):
    df = pd.concat([X, y, df_scanner], axis=1).dropna(axis=0)
    df = df[df["scanner_label"].isin([0, 1, 2, 5, 6])]
    #Harmonization step:
    data = df.loc[:, [
        k for k in df.columns if k not in ["scanner_label", "plc_status"]
    ]]
    covars = df.loc[:, ["scanner_label"]]
    data_combat = neuroCombat(dat=data.T,
                              covars=covars,
                              batch_col="scanner_label")["data"]
    return data_combat.T, df["plc_status"].values


if __name__ == '__main__':
    main()
