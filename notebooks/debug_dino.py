from pathlib import Path
import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.model_dino.tf_data import get_tf_data, RandomStandardization, GaussianBlur
from src.model_dino.models import DinoModel
from src.models.models import unet_model

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
h5_file = h5py.File(
    "/home/valentin/python_wkspce/plc_segmentation/data/processed/hdf5_2d/data_selected_slices.hdf5",
    "r")
clinical_df = pd.read_csv(
    "/home/valentin/python_wkspce/plc_segmentation/data/clinical_info.csv"
).set_index("patient_id")

ds = get_tf_data(h5_file, clinical_df, local_inpainting=False, n_channels=2)

model_s = unet_model(10,
                     last_activation="softmax",
                     pretrained=False,
                     input_shape=(None, None, 2))
model_t = unet_model(10,
                     last_activation="softmax",
                     pretrained=False,
                     input_shape=(None, None, 2))

model = DinoModel(model_s, model_t)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), run_eagerly=True)

model.fit(x=ds.batch(4).take(1))
