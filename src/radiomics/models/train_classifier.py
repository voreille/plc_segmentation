import os
import random

from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from src.radiomics.models.utils import RemoveHighlyCorrelatedFeatures
from src.data.utils import get_split

os.environ['PYTHONHASHSEED'] = '0'
random.seed(12345)
seed(1)


def main(path_to_features,
         path_to_outcomes,
         pipe,
         param_grid,
         voi="GTV_L",
         modality="CT",
         shuffle=True):

    df = load_data(path_to_features, path_to_outcomes)
    X_train, X_test, y_train, y_test, feature_names = get_formatted_data(
        df, split=0, modality=modality, voi=voi)

    if shuffle:
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx, :]
        y_train = y_train[idx]

        idx = np.arange(X_test.shape[0])
        np.random.shuffle(idx)
        X_test = X_test[idx, :]
        y_test = y_test[idx]


    search = GridSearchCV(pipe,
                          param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=23,
                          refit=True,
                          verbose=1,
                          scoring="roc_auc")

    search.fit(X_train, y_train)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    y_test_oh = y_test.reshape(-1, 1)
    y_test_oh = enc.fit_transform(y_test_oh)

    plot_roc_curve(X_test, y_test_oh, search.best_estimator_)
    # storing the scores, we can easily extend this to
    # all the usefule metrics
    n_bootstraps = 1000

    score = pd.DataFrame()
    for k in range(n_bootstraps):
        X_test_resampled, y_test_resampled = resample(
            X_test, y_test, replace=True, n_samples=len(y_test))


        y_pred = search.predict(X_test_resampled)
        y_score = search.predict_proba(X_test_resampled)[:, 1]
        score = append_score(score, y_test_resampled, y_pred, y_score, pos_label=1)

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
    selected_features = feature_names[
        search.best_estimator_['feature_selection'].get_support()]
    print(f"Featuring these features: {selected_features}")
    print(f"With the best model {search.best_estimator_}")
    print("voila")


def print_results(d):
    for key, item in d.items():
        print(f"{key}: {item[0]:0.2f} ({item[1]:0.2f} - {item[2]:0.2f})")


def plot_roc_curve(x_test, y_test, model, name_fig="roc_auc_best_model_pt"):
    n_classes = y_test.shape[1]

    # Learn to predict each class against the other
    y_score = model.predict_proba(x_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 1
    plt.plot(fpr[1],
             tpr[1],
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(f"reports/figures/{name_fig}.png", dpi=200)


def clean_df(df):
    # df = df.drop(['Image', 'Mask', 'patient', 'class'], axis=1)
    col_to_drop = [c for c in df.columns if c.startswith('diagnostics')]
    col_to_drop.extend(
        [c for c in df.columns if 'glcm' in c and 'original' not in c])
    # col_to_drop.extend([c for c in df.columns if 'glcm' in c])
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
    # col_to_drop.extend([c for c in df.columns if 'sigma-4-0-mm' in c])
    # col_to_drop.extend([c for c in df.columns if 'sigma-3-0-mm' in c])
    # col_to_drop.extend([c for c in df.columns if 'sigma-5-0-mm' in c])
    # col_to_drop.extend(
    #     [c for c in df.columns if 'original' not in c or 'Maximum' not in c])

    return df.drop(col_to_drop, axis=1)


def load_data(path_to_features, path_to_outcomes):
    df = pd.read_csv(path_to_features)
    clinical_df = pd.read_csv(path_to_outcomes).set_index("patient_id")
    df["plc_status"] = df["patient_id"].map(
        lambda x: clinical_df.loc[x, "plc_status"])

    df = df.drop(["Unnamed: 0"], axis=1)
    df = clean_df(df)

    return df


def get_formatted_data(df, split=0, modality="CT", voi="GTV_L"):
    ids_train, ids_val, ids_test = get_split(split)
    ids_train.extend(ids_val)
    df = df[(df["modality"] == modality) & (df["voi"] == voi)]
    df = df.set_index("patient_id")
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


if __name__ == '__main__':

    path_to_features = "/home/valentin/python_wkspce/plc_segmentation/data/processed/radiomics/extracted_features.csv"
    path_to_outcomes = "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info.csv"

    scaler = StandardScaler()

    # clf_lr = LogisticRegression(penalty="none",
    #                             solver='sag',
    #                             max_iter=1000)
    clf_lr = LogisticRegression(penalty='l2',
                                C=100000,
                                solver='liblinear',
                                max_iter=1000)
    clf_lr_l1 = LogisticRegression(penalty='l1', solver='liblinear')
    clf_rf = RandomForestClassifier()
    # clf = DummyClassifier(strategy='most_frequent')

    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_selection', None),  # SelectKBest(f_classif)),
        ('classifier', None),
    ])

    F_OPTIONS = [3, 4, 5]  # "all"]
    # F_OPTIONS = [1, 3, 5]  # "all"]
    # F_OPTIONS = [1]  # "all"]
    C_OPTIONS = [10**k for k in range(-4, 5, 1)]
    K_OPTIONS = [k for k in range(1, 11)]

    param_grid = [
        #     {
        #         'feature_selection': [SelectKBest(f_classif)],
        #         'feature_selection__k': F_OPTIONS,
        #         'classifier': [clf_rf],
        #         'classifier__n_estimators': [100, 150]
        #     },
        {
            'feature_selection': [SelectKBest(f_classif)],
            'feature_selection__k': F_OPTIONS,
            'classifier': [clf_lr],
        },
        # {
        #     'feature_selection': [SelectKBest(f_classif)],
        #     'feature_selection__k': F_OPTIONS,
        #     'classifier': [clf_lr_l1],
        #     'classifier__C': C_OPTIONS,
        # },
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

    main(path_to_features, path_to_outcomes, pipe, param_grid)
