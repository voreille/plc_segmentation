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
from sklearn.model_selection import GridSearchCV, train_test_split
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
from tqdm import tqdm

from src.radiomics.utils import RemoveHighlyCorrelatedFeatures
from src.radiomics.data import load_data, get_formatted_data

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)

project_dir = Path(__file__).resolve().parents[2]
path_to_features = project_dir / "data/processed/radiomics/extracted_features.csv"
path_to_outcomes = project_dir / "data/clinical_info.csv"

model_type = "default"
n_splits = 10
n_repeats = 10
modalities = ["CT", "PT"]
# vois = ["GTV_L"]
vois = ["GTV_L", "GTV_T", "GTV_N"]

store_dummy = False
store_standard = True
store_suvmax = False
store_combat = False
store_fusion = False
store_size_analysis = False


def main():
    df = load_data(path_to_features,
                   path_to_outcomes,
                   clinical_info=["plc_status", "is_chuv"])
    rkf = RepeatedStratifiedKFold(n_splits=n_splits,
                                  n_repeats=n_repeats,
                                  random_state=4)
    if store_standard:
        scores = list()
        feature_counts = list()
        for modality, voi in product(modalities, vois):
            s, fc = compute_rkf(
                rkf,
                df,
                modality=modality,
                voi=voi,
                model=model_dict[model_type](),
                return_features_counts=True,
            )
            scores.append(s)
            feature_counts.append(fc)

        scores = pd.concat(scores, ignore_index=True)
        feature_counts = pd.concat(feature_counts, ignore_index=True)
        scores.to_csv(project_dir /
                      f"results/radiomics/rkf_results/{model_type}.csv")
        feature_counts.to_csv(
            project_dir /
            f"results/radiomics/rkf_results/{model_type}_feature_counts.csv")

    if store_fusion:
        scores = list()
        for voi in vois:
            s = compute_rkf_fusion(
                rkf,
                df,
                voi=voi,
                model_ct=model_dict[model_type](),
                model_pt=model_dict[model_type](),
                model_fusion=get_simplest_model(),
            )
            scores.append(s)

        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(project_dir /
                      f"results/radiomics/rkf_results/{model_type}_fusion.csv")

    if store_size_analysis:
        scores = list()
        for voi in vois:
            s = compute_size_analysis(
                rkf,
                df,
                voi=voi,
                model_ct=model_dict[model_type](),
                model_pt=model_dict[model_type](),
                model_fusion=get_simplest_model(),
            )
            scores.append(s)

        scores = pd.concat(scores, ignore_index=True)
        scores.to_csv(
            project_dir /
            f"results/radiomics/rkf_results/{model_type}_fusion_size_analysis.csv"
        )

    if store_combat:
        df_scanner = pd.read_csv(
            "/home/valentin/python_wkspce/plc_segmentation/data/scanner_labels.csv"
        ).set_index("patient_id")
        scores = list()
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
                compute_rkf_suvmax(rkf, df, voi="GTV_L"),
                compute_rkf_suvmax(rkf, df, voi="GTV_N"),
                compute_rkf_suvmax(rkf, df, voi="GTV_T"),
            ],
            ignore_index=True,
        )

        scores.to_csv(project_dir / "results/radiomics/rkf_results/suvmax.csv")


def compute_size_analysis(rkf,
                          df,
                          voi="GTV_L",
                          model_ct=None,
                          model_pt=None,
                          model_fusion=None):
    X_pt, y, _ = get_formatted_data(df, modality="PT", voi=voi)
    X_ct, _, _ = get_formatted_data(df, modality="CT", voi=voi)

    score = pd.DataFrame()
    for r_train in tqdm([0.1 * k for k in range(2, 11)],
                        desc="Running for different training size"):
        for i, (idx_train, idx_test) in enumerate(rkf.split(X_pt, y=y)):
            y_train, y_test = y[idx_train], y[idx_test]
            if r_train != 1.0:
                idx_train, _ = train_test_split(idx_train,
                                                train_size=r_train,
                                                stratify=y_train)
                y_train = y[idx_train]

            X_train_pt, X_test_pt = X_pt[idx_train], X_pt[idx_test]
            X_train_ct, X_test_ct = X_ct[idx_train], X_ct[idx_test]

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
    score["modality"] = "PET/CT"
    score["voi"] = voi
    return score


def compute_rkf_fusion(rkf,
                       df,
                       voi="GTV_L",
                       model_ct=None,
                       model_pt=None,
                       model_fusion=None):
    X_pt, y, _ = get_formatted_data(df, modality="PT", voi=voi)
    X_ct, _, _ = get_formatted_data(df, modality="CT", voi=voi)

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
    score["modality"] = "PET/CT"
    score["voi"] = voi
    return score


def compute_rkf_suvmax(rkf, df, voi="GTV_L"):
    modality = "SUVmax"
    X_gtvl, y, feature_names_gtvl = get_formatted_data(df,
                                                       modality="PT",
                                                       voi=voi)
    max_indice = np.where(
        np.array(feature_names_gtvl) == "original_firstorder_Maximum")[0]

    X = X_gtvl[:, max_indice]

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
    score["modality"] = modality
    score["voi"] = voi
    return score


def get_selected_features(model):
    x = np.arange(model.n_features_in_)
    x = x[np.newaxis, ...]
    x = model.best_estimator_[1:-1].transform(x)
    support = np.zeros((model.n_features_in_, ))
    support[x[0, :]] = 1
    return support


def compute_rkf(rkf,
                df,
                modality="CT",
                voi="GTV_L",
                model=None,
                return_features_counts=True,
                df_scanner=None):
    X, y = get_formatted_data(df, modality=modality, voi=voi, return_df=True)

    if return_features_counts:
        features = X.columns
        features_count = np.zeros((len(features), ))

    if df_scanner is not None:
        X, y = apply_combat(X, y, df_scanner)
    else:
        X, y = X.values, y.values

    score = pd.DataFrame()
    for i, (idx_train, idx_test) in enumerate(rkf.split(X, y=y)):

        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        model.fit(X_train, y_train)
        if return_features_counts:
            features_count += get_selected_features(model)

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
    if return_features_counts:
        features_count_df = pd.DataFrame({
            "feature_name": features,
            "feature_count": features_count
        })
        features_count_df["modality"] = modality
        features_count_df["voi"] = voi
        return score, features_count_df
    else:
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
             penalty='none',
             solver='newton-cg',
             max_iter=1000,
         )),
    ])


def get_default_model():
    scaler = StandardScaler()

    clf_lr = LogisticRegression(penalty='none',
                                solver='newton-cg',
                                max_iter=1000)
    clf_lr_l1 = LogisticRegression(penalty='l1', solver='liblinear')
    clf_rf = RandomForestClassifier()

    rmf = RemoveHighlyCorrelatedFeatures()
    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_removal', rmf),
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
        # {
        #     'feature_selection': [PCA()],
        #     'feature_selection__n_components': K_OPTIONS,
        #     'classifier': [clf_lr],
        # },
        # {
        #     'feature_selection': [PCA()],
        #     'feature_selection__n_components': K_OPTIONS,
        #     'classifier': [clf_lr_l1],
        #     'classifier__C': C_OPTIONS,
        # },
        # {
        #     'feature_selection': [PCA()],
        #     'feature_selection__n_components': K_OPTIONS,
        #     'classifier': [clf_rf],
        # },
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
    clf_lr = LogisticRegression(penalty='none',
                                solver='newton-cg',
                                max_iter=1000)

    rmf = RemoveHighlyCorrelatedFeatures()
    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_removal', rmf),
        ('feature_selection', None),  # SelectKBest(f_classif)),
        ('classifier', None),
    ])

    F_OPTIONS = [1, 3, 5, 10, "all"]
    # F_OPTIONS = [3, 4, 5]

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
