from pathlib import Path

from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from radiomics.featureextractor import RadiomicsFeatureExtractor

from src.models.utils import predict_volume, reshape_image_unet
from src.models.models import unet_model
from src.data.tf_data_hdf5 import preprocess_image

project_dir = Path(__file__).resolve().parents[2]

model_path = Path(
    "/home/valentin/python_wkspce/plc_segmentation/models/unet__a_0.75__upsmpl_upsampling__split_0__ovrsmpl_True__con_nothing20211214-104527/"
)
models_dir = project_dir / "models"


def append_results(patient_id, result, df):
    output = {
        key: item
        for key, item in result.items() if "diagnostics" not in key
    }
    output.update({"patient_id": patient_id})

    return df.append(output, ignore_index=True)


def extract_radiomics(model,
                      patient_list,
                      h5_file,
                      ct_clipping=[-1350, 150],
                      config_ct=None,
                      config_pt=None):
    if config_ct:
        params_ct = str(config_ct)
    else:
        params_ct = str(project_dir / "src/features/param_CT.yaml")

    if config_pt:
        params_pt = str(config_pt)
    else:
        params_pt = str(project_dir / "src/features/param_PT.yaml")

    extractor_ct = RadiomicsFeatureExtractor(params_ct)
    extractor_pt = RadiomicsFeatureExtractor(params_pt)
    total_results_ct = pd.DataFrame()
    total_results_pt = pd.DataFrame()
    for p in tqdm(patient_list):
        image = h5_file[p]["image"][()]
        mask = h5_file[p]["mask"][()]
        
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

    return total_results_ct, total_results_pt
