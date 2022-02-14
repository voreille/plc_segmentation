import os
import random

from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)

PATH_TO_FEATURES = "/home/valentin/python_wkspce/plc_segmentation/data/processed/radiomics/extracted_features.csv"
PATH_OUTCOMES = "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info.csv"
MODALITY = "CT"
VOI = "GTV_L"
N_BOOTSTRAP = 1000


def main():
    df = load_data(PATH_TO_FEATURES, PATH_OUTCOMES)
    X_train, X_test, y_train, y_test, feature_names = get_formatted_data(
        df,
        modality=MODALITY,
        voi=VOI,
        test_size=0.4,
    )

    search = get_gridsearch()

    search.fit(X_train, y_train)

    # bootstrap
    score = pd.DataFrame()
    for _ in range(N_BOOTSTRAP):
        X_test_resampled, y_test_resampled = resample(
            X_test,
            y_test,
            replace=True,
            n_samples=len(y_test),
            stratify=y_test,
        )

        y_pred = search.predict(X_test_resampled)
        y_score = search.predict_proba(X_test_resampled)[:, 1]
        score = append_score(score,
                             y_test_resampled,
                             y_pred,
                             y_score,
                             pos_label=1)

    ic_score = {
        'roc_auc': [],
        'specificity': [],
        'sensitivity': [],
        'accuracy': [],
        'precision': [],
        'npv': []
    }

    for col in score.columns:
        ic_score[col].append(np.mean(score[col].values))
        ic_score[col].append(np.percentile(score[col].values, 2.5))
        ic_score[col].append(np.percentile(score[col].values, 97.5))

    print_results(ic_score)

    # finding selected features does not work with PCA
    try:
        selected_features = feature_names[get_features_support(search) != 0]
        print(f"Featuring these features: {selected_features}")
    except IndexError as e:
        print("Printing selected features from PCA is not supported")

    print(f"With the best model {search.best_estimator_}")
    print("voila")


def print_results(d):
    for key, item in d.items():
        print(f"{key}: {item[0]:0.2f} ({item[1]:0.2f} - {item[2]:0.2f})")


def clean_df(df):
    col_to_drop = [c for c in df.columns if c.startswith('diagnostics')]
    col_to_drop.extend(
        [c for c in df.columns if 'glcm' in c and 'original' not in c])
    col_to_drop.extend([c for c in df.columns if 'RootMeanSquared' in c])
    col_to_drop.extend(
        [c for c in df.columns if '_MeanAbsoluteDeviation' in c])
    col_to_drop.extend([c for c in df.columns if '_Median' in c])
    col_to_drop.extend([c for c in df.columns if '_Range' in c])
    col_to_drop.extend([c for c in df.columns if '_InterquartileRange' in c])
    col_to_drop.extend([c for c in df.columns if 'Percentile' in c])
    col_to_drop.extend([c for c in df.columns if '_RootMeanSquared' in c])
    col_to_drop.extend([c for c in df.columns if '_TotalEnergy' in c])
    col_to_drop.extend([c for c in df.columns if '_Uniformity' in c])
    col_to_drop.extend([c for c in df.columns if 'wavelet' in c])
    col_to_drop.extend([c for c in df.columns if 'shape' in c])

    return df.drop(col_to_drop, axis=1)


def load_data(path_to_features, path_to_outcomes):
    df = pd.read_csv(path_to_features)
    clinical_df = pd.read_csv(path_to_outcomes).set_index("patient_id")
    df["plc_status"] = df["patient_id"].map(
        lambda x: clinical_df.loc[x, "plc_status"])

    df = df.drop(["Unnamed: 0"], axis=1)
    df = clean_df(df)

    return df


def get_formatted_data(df, modality="CT", voi="GTV_L", test_size=0.2):
    df = df[(df["modality"] == modality) & (df["voi"] == voi)]
    ids = df["patient_id"].values
    df = df.set_index("patient_id")
    ids_train, ids_test = train_test_split(ids,
                                           test_size=test_size,
                                           shuffle=True,
                                           stratify=df.loc[ids, "plc_status"])
    outcomes_df = df["plc_status"]
    df = df.drop(["plc_status", "voi", "modality"], axis=1)

    X_train = df.loc[ids_train, :].values
    X_test = df.loc[ids_test, :].values
    feature_names = df.columns

    y_train = outcomes_df.loc[ids_train].values
    y_test = outcomes_df.loc[ids_test].values

    return X_train, X_test, y_train, y_test, feature_names


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


def get_features_support(fitted_gridsearch):
    x = np.arange(fitted_gridsearch.n_features_in_)
    x = x[np.newaxis, ...]
    x = fitted_gridsearch.best_estimator_[1:-1].transform(x)
    support = np.zeros((fitted_gridsearch.n_features_in_, ))
    support[x[0, :]] = 1
    return support


def get_gridsearch():
    scaler = StandardScaler()

    clf_lr = LogisticRegression(penalty="none", solver='sag', max_iter=1000)
    clf_rf = RandomForestClassifier()

    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_selection', None),
        ('classifier', None),
    ])

    F_OPTIONS = [1, 2, 3, 5]
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
            'feature_selection': [PCA()],
            'feature_selection__n_components': K_OPTIONS,
            'classifier': [clf_lr],
        },
        {
            'feature_selection': [PCA()],
            'feature_selection__n_components': K_OPTIONS,
            'classifier': [clf_rf],
        },
    ]

    return GridSearchCV(pipe,
                        param_grid,
                        cv=StratifiedKFold(),
                        n_jobs=23,
                        refit=True,
                        verbose=1,
                        scoring="roc_auc")


if __name__ == '__main__':
    main()
