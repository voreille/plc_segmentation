from pathlib import Path
import os

import h5py
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from src.models.utils import predict_volume, reshape_image_unet
from src.data.tf_data_hdf5 import RandomStandardization
from src.data.utils import get_split

project_dir = Path(__file__).resolve().parents[2]

model_path = Path(
    "/home/valentin/python_wkspce/plc_segmentation/models/unet__prtrnd_True__a_0.25__wt_1.0__wl_0.0__wgtvl_1.0upsmpl_upsampling__split_0__ovrsmpl_True__con_specialltyp_sum_of_dice__mltsk_False__20220131-210226/model_weight"
)
models_dir = project_dir / "models"
params_ct = str(project_dir / "src/features/param_CT.yaml")
params_pt = str(project_dir / "src/features/param_PT.yaml")

split = 0
gpu_id = '2'
only_volume = True
adjust_input_shape = True


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(
        project_dir / "data/processed/hdf5_2d/data_selected_slices.hdf5", "r")

    model = tf.keras.models.load_model(model_path, compile=False)
    output_name = str(model_path).split('/')[-2]
    preprocessor = RandomStandardization(p=0.0)

    dir_to_save = project_dir / "data/predictions" / output_name
    dir_to_save.mkdir(exist_ok=True)
    ids_train, ids_val, ids_test = get_split(split)

    dir_test = dir_to_save / "test"
    dir_test.mkdir(exist_ok=True)
    save_pred(ids_test, model, h5_file, dir_test, preprocessor=preprocessor)

    dir_train = dir_to_save / "train"
    dir_train.mkdir(exist_ok=True)
    save_pred(ids_train, model, h5_file, dir_train, preprocessor=preprocessor)

    dir_val = dir_to_save / "val"
    dir_val.mkdir(exist_ok=True)
    save_pred(ids_val, model, h5_file, dir_val, preprocessor=preprocessor)

    h5_file.close()


def to_sitk(image):
    return sitk.GetImageFromArray(np.transpose(image, (2, 1, 0)))


def save_pred(
    patient_list,
    model,
    h5_file,
    dir_to_save,
    preprocessor=None,
    n_channels=3,
):
    for p in tqdm(patient_list):
        image = h5_file[p]["image"][()]
        mask = h5_file[p]["mask"][()]
        image = reshape_image_unet(image, mask[..., 2] + mask[..., 3])
        mask = reshape_image_unet(mask, mask[..., 2] + mask[..., 3])
        if n_channels == 3:
            image = np.stack(
                [image[..., 0], image[..., 1],
                 np.zeros_like(image[..., 0])],
                axis=-1)
        image = preprocessor(image)
        prediction = predict_volume(image, model)
        prediction = prediction[:, :, :, 1]

        image_ct = to_sitk(image[:, :, :, 0])
        image_pt = to_sitk(image[:, :, :, 1])
        gtvl_pred = to_sitk(prediction)
        gtvt_gt = to_sitk(mask[:, :, :, 0])
        gtvl_gt = to_sitk(mask[:, :, :, 1])
        sitk.WriteImage(image_ct, str(dir_to_save / (p + "_CT.nii.gz")))
        sitk.WriteImage(image_pt, str(dir_to_save / (p + "_PT.nii.gz")))
        sitk.WriteImage(gtvt_gt, str(dir_to_save / (p + "_GTVT_gt.nii.gz")))
        sitk.WriteImage(gtvl_gt, str(dir_to_save / (p + "_GTVL_gt.nii.gz")))
        sitk.WriteImage(gtvl_pred,
                        str(dir_to_save / (p + "_GTVL_pred.nii.gz")))


if __name__ == '__main__':
    main()