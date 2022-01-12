from pathlib import Path

import json
import h5py
import pandas as pd

from src.data.utils import generate_split

project_dir = Path(__file__).resolve().parents[2]
n_rep = 20
output_path = project_dir / "data/splits.json"


def main():
    clinical_df = pd.read_csv(
        "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info.csv"
    ).set_index("patient_id")

    file = h5py.File(
        "/home/valentin/python_wkspce/plc_segmentation/data/processed/hdf5_2d/data.hdf5",
        "r")
    ids_list = list()
    patient_list = list(file.keys())
    file.close()

    # remove dupplicata, plc status is ok, but chuv status is not
    patient_list.remove("PatientLC_63")  # Just one lung
    patient_list.remove("PatientLC_71")  # the same as 69
    patient_list.remove("PatientLC_21")  # the same as 20 

    for k in range(n_rep):
        ids_train, ids_val, ids_test = generate_split(patient_list, clinical_df)
        ids_list.append({
            "train": ids_train,
            "val": ids_val,
            "test": ids_test,
        })

    with open(output_path, "w") as f:
        json.dump(ids_list, f)


if __name__ == '__main__':
    main()