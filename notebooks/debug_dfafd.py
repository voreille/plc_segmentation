# %%
from itertools import product

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from src.radiomics.models.data import load_data, get_formatted_data
from src.radiomics.models.utils import RemoveHighlyCorrelatedFeatures

# %%

clinical_df = pd.read_csv(
    "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info_updated.csv"
).set_index("patient_id")

# %%
ids = clinical_df.index
ids = [
    i for i in ids if i not in
    ["PatientLC_71", "PatientLC_21", "PatientLC_63", "PatientLC_72"]
]
clinical_df = clinical_df.loc[ids, :]

# %%
clinical_df.shape


# %%
def get_gridsearch():
    scaler = StandardScaler()

    clf_lr = LogisticRegression(penalty="none", solver='sag', max_iter=1000)
    clf_rf = RandomForestClassifier()

    pipe = Pipeline(steps=[
        ('normalization', scaler),
        ('feature_removal', RemoveHighlyCorrelatedFeatures()),
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


# %%
search = get_gridsearch()

# %%
df = load_data(
    "data/processed/radiomics/extracted_features.csv",
    # "../data/processed/radiomics/extracted_features_auto.csv",
    "data/clinical_info_updated.csv",
)

# %%
X, y = get_formatted_data(df, modality="PT", voi="GTV_L", return_df=True)

# %%
X.shape

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# %%
def get_scores(*, search, skf, modality, voi):
    X, y = get_formatted_data(df, modality=modality, voi=voi, return_df=True)
    y_output = y.copy()
    y_output["y_pred"] = 0
    y_output["y_pred_proba"] = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = np.squeeze(y.iloc[train_index].values)

        search.fit(X_train, y_train)

        y_output.iloc[test_index,
                      y_output.columns.get_loc("y_pred")] = np.squeeze(
                          search.predict(X_test))
        y_output.iloc[test_index,
                      y_output.columns.get_loc("y_pred_proba")] = np.squeeze(
                          search.predict_proba(X_test)[:, 1])

    return y_output


# %%
def get_scores_fusion(*, skf, voi):
    X_pt, _ = get_formatted_data(df, modality="PT", voi=voi, return_df=True)
    X_ct, y = get_formatted_data(df, modality="CT", voi=voi, return_df=True)
    y_output = y.copy()
    y_output["y_pred"] = 0
    y_output["y_pred_proba"] = 0
    search_pt = get_gridsearch()
    search_ct = get_gridsearch()
    model_fusion = LogisticRegression(penalty="none",
                                      solver='sag',
                                      max_iter=1000)
    for train_index, test_index in skf.split(X, y):
        X_train_pt, X_test_pt = X_pt.iloc[train_index], X_pt.iloc[test_index]
        X_train_ct, X_test_ct = X_ct.iloc[train_index], X_ct.iloc[test_index]
        y_train = np.squeeze(y.iloc[train_index].values)

        search_pt.fit(X_train_pt, y_train)
        search_ct.fit(X_train_ct, y_train)
        model_fusion.fit(
            np.stack(
                [
                    search_pt.predict_proba(X_train_pt)[:, 1],
                    search_ct.predict_proba(X_train_ct)[:, 1],
                ],
                axis=1,
            ), y_train)

        def make_prediction(x_pt, x_ct):
            preds = np.stack([
                search_pt.predict_proba(x_pt)[:, 1],
                search_ct.predict_proba(x_ct)[:, 1],
            ],
                             axis=1)
            return model_fusion.predict(preds), model_fusion.predict_proba(
                preds)[:, 1]

        predictions, predictions_proba = make_prediction(X_test_pt, X_test_ct)
        y_output.iloc[test_index,
                      y_output.columns.get_loc("y_pred")] = predictions
        y_output.iloc[
            test_index,
            y_output.columns.get_loc("y_pred_proba")] = predictions_proba

    return y_output


# %%
def compute_all_mcnemar(x1, x2):
    cm = confusion_matrix(x1, x2)
    return {
        "confusion_matrix": cm,
        "pvalue": mcnemar(cm, exact=False, correction=False),
        "pvalue_corrected": mcnemar(cm, exact=False, correction=True),
        "pvalue_exact": mcnemar(cm, exact=True, correction=False),
    }


# %%
output_1 = get_scores_fusion(skf=skf, voi="GTV_L")
print(output_1)
