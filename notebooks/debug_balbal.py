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
from src.roc_comparison.compare_auc_delong_xu import delong_roc_test



clinical_df = pd.read_csv("/home/valentin/python_wkspce/plc_segmentation/data/clinical_info_updated.csv").set_index("patient_id")

ids = clinical_df.index
ids = [
    i for i in ids if i not in
    ["PatientLC_71", "PatientLC_21", "PatientLC_63", "PatientLC_72"]
]
clinical_df = clinical_df.loc[ids, :]



df  = load_data(
    "data/processed/radiomics/extracted_features.csv",
    # "../data/processed/radiomics/extracted_features_auto.csv",
    "data/clinical_info_updated.csv",
)
df = get_formatted_data(df, modality="CT", voi="GTV_L", return_df=True)
print(df)




