from pathlib import Path
import os

import h5py
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
import numpy as np
from radiomics.featureextractor import RadiomicsFeatureExtractor

from src.models.utils import predict_volume
from src.models.models import unet_model
from src.data.tf_data_hdf5 import get_bb_mask_voxel, preprocess_image

project_dir = Path(__file__).resolve().parents[2]
# model_path = ("/home/valentin/python_wkspce/plc_segmentation"
#               "/models/pretrained_unet__alpha_0.25__upsampling_upsampling__"
#               "split_0__oversample_True__rangle_"
#               "None__rshift_None__20211211-224814/model_weight")
# model_path = "GroundTruth"
models_path = project_dir / "models"
params_ct = str(project_dir / "src/features/param_CT.yaml")
params_pt = str(project_dir / "src/features/param_PT.yaml")

split = 0
gpu_id = '0'
only_volume = True
adjust_input_shape = True
ct_clipping = [-1350, 150]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(project_dir / "data/processed/hdf5_2d/data.hdf5", "r")
    patient_list = list(h5_file)
    patient_list.remove("PatientLC_63")  # Just one lung
    patient_list.remove("PatientLC_72")  # the same as 70

    for model_path in tqdm(models_path.iterdir()):
        if only_volume:
            process_one_model_only_volume(
                str(model_path / "model_weight"),
                patient_list,
                h5_file,
            )
        else:
            process_one_model(
                str(model_path / "model_weight"),
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


def reshape_image_unet(image, mask_lung, level=5, p_id=""):
    bb_lung = get_bb_mask_voxel(mask_lung)
    center = ((bb_lung[:3] + bb_lung[3:]) // 2).astype(int)
    lung_shape = np.abs(bb_lung[3:] - bb_lung[:3])
    max_shape = np.max(lung_shape[:2])
    final_shape = max_shape + 2**level - max_shape % 2**level
    radius = int(final_shape // 2)
    image_cropped = image[center[0] - radius:center[0] + radius,
                          center[1] - radius:center[1] + radius, :, :]
    min_shape = np.min(image_cropped.shape[:2])
    if min_shape < final_shape:  # Maybe do some recursion
        final_shape = min_shape - min_shape % 2**level
        print(
            f"THE PATIENT {p_id} has some weird shape going on: {image.shape}")

        radius = int(final_shape // 2)
        image_cropped = image[center[0] - radius:center[0] + radius,
                              center[1] - radius:center[1] + radius, :, :]

    return image_cropped


if __name__ == '__main__':
    main()