from pathlib import Path
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

project_dir = Path(__file__).resolve().parents[2]

model_path = Path(
    "/home/valentin/python_wkspce/plc_segmentation/models/unet__a_0.75__upsmpl_upsampling__split_0__ovrsmpl_True__con_nothing20211214-104527/"
)
models_dir = project_dir / "models"
params_ct = str(project_dir / "src/features/param_CT.yaml")
params_pt = str(project_dir / "src/features/param_PT.yaml")

split = 0
gpu_id = '0'
only_volume = True
adjust_input_shape = True
ct_clipping = (-1350, 150)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(project_dir / "data/processed/hdf5_2d/data.hdf5", "r")
    patient_list = list(h5_file)
    patient_list.remove("PatientLC_63")  # Just one lung
    patient_list.remove("PatientLC_72")  # the same as 70
    if model_path is None:
        model_paths = [
            p for p in models_dir.iterdir() if "oversample_False" in p.name
        ]
    else:
        model_paths = [model_path]
    for path in tqdm(model_paths):
        if only_volume:
            process_one_model_only_volume(
                str(path / "model_weight"),
                patient_list,
                h5_file,
            )
        else:
            process_one_model(
                str(path / "model_weight"),
                patient_list,
                h5_file,
            )

    h5_file.close()


def process_one_model_only_volume(model_path, patient_list, h5_file):
    if adjust_input_shape:
        model = unet_model(3)
        model.set_weights(
            tf.keras.models.load_model(model_path,
                                       compile=False).get_weights())
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
    output_name = model_path.split('/')[-2]

    total_results = pd.DataFrame()
    for p in tqdm(patient_list):
        image = h5_file[p]["image"][()]
        mask = h5_file[p]["mask"][()]
        image = reshape_image_unet(image, mask[..., 2] + mask[..., 3])
        image = preprocess_image(image, ct_clipping=ct_clipping)
        prediction = predict_volume(image, model)
        prediction = (prediction[:, :, :, 1] > 0.5).astype(int)
        volume = np.sum(prediction)
        total_results = total_results.append(
            {
                "patient_id": p,
                "volume": volume,
            },
            ignore_index=True,
        )

    dir_to_save = project_dir / "results" / output_name
    dir_to_save.mkdir(exist_ok=True)
    total_results.to_csv(dir_to_save / "only_volume.csv")


def process_one_model(model_path, patient_list, h5_file):
    if model_path != "GroundTruth":
        model = tf.keras.models.load_model(model_path, compile=False)
        output_name = model_path.split('/')[-2]
    else:
        output_name = "ground_truth"

    extractor_ct = RadiomicsFeatureExtractor(params_ct)
    extractor_pt = RadiomicsFeatureExtractor(params_pt)
    total_results_ct = pd.DataFrame()
    total_results_pt = pd.DataFrame()
    for p in tqdm(patient_list):
        image = h5_file[p]["image"][()]
        mask = h5_file[p]["mask"][()]
        if model_path == "GroundTruth":
            prediction = h5_file[p]["mask"][()]
            prediction = prediction[..., 1]
        else:
            image = reshape_image_unet(image, mask[..., 2] + mask[..., 3])
            image = preprocess_image(image, ct_clipping=ct_clipping)
            prediction = predict_volume(image, model)
            prediction = (prediction[:, :, :, 1] > 0.5).astype(int)

        image_ct = sitk.GetImageFromArray(image[:, :, :, 0])
        image_pt = sitk.GetImageFromArray(image[:, :, :, 1])
        gtvl = sitk.GetImageFromArray(prediction)

        results_ct = extractor_ct.execute(image_ct, gtvl)
        results_pt = extractor_pt.execute(image_pt, gtvl)
        total_results_ct = append_results(p, results_ct, total_results_ct)
        total_results_pt = append_results(p, results_pt, total_results_pt)
    total_results_ct.to_csv(project_dir / "results" / output_name /
                            "features_CT.csv")
    total_results_pt.to_csv(project_dir / "results" / output_name /
                            "features_PT.csv")


def append_results(patient_id, result, df):
    output = {
        key: item
        for key, item in result.items() if "diagnostics" not in key
    }
    output.update({"patient_id": patient_id})

    return df.append(output, ignore_index=True)


if __name__ == '__main__':
    main()