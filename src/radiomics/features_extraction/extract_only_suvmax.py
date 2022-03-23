from pathlib import Path
from itertools import product
from multiprocessing import Pool
import os

import h5py
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
from radiomics.featureextractor import RadiomicsFeatureExtractor

from src.models.utils import predict_volume, reshape_image_unet
from src.models.models import unet_model
from src.data.tf_data_hdf5 import preprocess_image

project_dir = Path(__file__).resolve().parents[3]

data_path = project_dir / "data/interim/nii_resampled"

params_ct = str(project_dir / "src/radiomics/parameters/param_CT_only1storder.yaml")
params_pt = str(project_dir / "src/radiomics/parameters/param_PT_only1storder.yaml")

vois = ["GTV_L", "GTV_T", "GTV_N"]

cores = 24


def main():
    patient_list = [
        f.name.split("__")[0] for f in data_path.rglob("*PT.nii.gz")
    ]

    extractors = {
        "CT": RadiomicsFeatureExtractor(params_ct),
        "PT": RadiomicsFeatureExtractor(params_pt),
    }

    extractor = Extractor(extractors)
    if cores == 1:
        results = list()
        for p in tqdm(patient_list):
            results.append(extractor(p))
    else:
        with Pool(cores) as pool:
            results = pool.map(
                extractor,
                patient_list,
            )

    df = pd.concat(results, axis=0)
    df.to_csv(project_dir / "data/processed/radiomics/extracted_features_only1storder.csv")


class Extractor():

    def __init__(
        self,
        extractors,
        modalities=None,
        vois=None,
    ):
        self.extractors = extractors
        if modalities is None:
            self.modalities = ["CT", "PT"]
        if vois is None:
            self.vois = ["GTV_L", "GTV_T", "GTV_N"]

    def __call__(self, patient):
        print(f"Processing patient {patient} - START")
        output_df = pd.DataFrame()
        for modality, voi in product(self.modalities, self.vois):
            image = sitk.ReadImage(
                str(data_path / f"{patient}__{modality}.nii.gz"))
            mask = sitk.ReadImage(
                str(data_path / f"{patient}__{voi}__RTSTRUCT__CT.nii.gz"))
            results = self.extractors[modality].execute(image, mask)

            output = {
                key: item
                for key, item in results.items() if "diagnostics" not in key
            }
            output.update({
                "patient_id": patient,
                "voi": voi,
                "modality": modality
            })
            output_df = output_df.append(output, ignore_index=True)
        print(f"Processing patient {patient} - DONE")
        return output_df


if __name__ == '__main__':
    main()