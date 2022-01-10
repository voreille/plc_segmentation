from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.utils import reshape_image_unet, predict_volume
from src.data.tf_data_hdf5 import preprocess_image


def evaluate_pred_volume(
    model,
    patient_list,
    h5_file,
    clinical_df,
    ct_clipping=[-1350, 150],
    n_channels=3,
):

    total_results = pd.DataFrame()
    for p in tqdm(patient_list):
        image = h5_file[p]["image"][()]
        mask = h5_file[p]["mask"][()]
        image = reshape_image_unet(image, mask[..., 2] + mask[..., 3])
        image = preprocess_image(image, ct_clipping=ct_clipping)
        image = image[..., :n_channels]
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

    total_results = total_results.set_index("patient_id")
    df = pd.concat([total_results, clinical_df["plc_status"]],
                   axis=1).dropna(axis=0)

    return roc_auc_score(df.loc[patient_list, "plc_status"],
                         df.loc[patient_list, "volume"])