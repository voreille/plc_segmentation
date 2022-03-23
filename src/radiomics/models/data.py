import pandas as pd
import numpy as np

from src.data.utils import get_split


def clean_df(df):
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


def load_data(path_to_features, path_to_outcomes, clinical_info=None):
    if clinical_info is None:
        clinical_info = ["plc_status"]
    df = pd.read_csv(path_to_features)
    clinical_df = pd.read_csv(path_to_outcomes)
    set_ids_1 = set(df["patient_id"].unique())
    set_ids_2 = set(clinical_df["patient_id"].unique())
    ids_list = set_ids_1.intersection(set_ids_2)
    df = df[df["patient_id"].isin(ids_list)]
    clinical_df = clinical_df[clinical_df["patient_id"].isin(ids_list)]
    clinical_df = clinical_df.set_index("patient_id")
    for col in clinical_info:
        df[col] = df["patient_id"].map(lambda x: clinical_df.loc[x, col])

    df = df.drop(["Unnamed: 0"], axis=1)
    df = clean_df(df)

    return df


def get_splitted_data(df, split=0, modality="CT", voi="GTV_L"):
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


def get_formatted_data(
    df,
    modality="CT",
    voi="GTV_L",
    return_df=False,
    outcome_key="plc_status",
    clinical_info=None,
):
    if clinical_info is None:
        clinical_info = []
    df = df[(df["modality"] == modality) & (df["voi"] == voi)]
    df = df.set_index("patient_id").sort_index()

    patient_list = [
        p for p in df.index if p not in
        ["PatientLC_63", "PatientLC_21", "PatientLC_71", "PatientLC_72"]
    ]
    outcomes_df = df[[outcome_key]]
    df = df.drop(["plc_status", "voi", "modality"], axis=1)
    col2rm = list((set([
        'patient_id', 'is_chuv', 'plc_status', 'pT', 'pN', 'M', 'Stage',
        'pathologic_type', 'new_cases', 'LymphangitisCT', 'sexe', 'age'
    ]) & set(df.columns)) - set(clinical_info))

    if col2rm:
        df = df.drop(col2rm, axis=1)

    df = df.loc[patient_list, :]
    outcomes_df = outcomes_df.loc[patient_list, :]

    if return_df:
        return df, outcomes_df
    else:
        return df.values, np.squeeze(outcomes_df.values), df.columns
