# %%
from pathlib import Path
import os
import time
from itertools import product

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_addons.optimizers import AdamW

from src.model_dino.tf_data import get_tf_data, RandomStandardization
from src.models.models import UnetLight, UnetLightDecorrelated
from src.data.utils import get_split

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
h5_file = h5py.File("data/processed/hdf5_2d/data_selected_slices.hdf5", "r")
clinical_df = pd.read_csv(
    "data/clinical_info_with_lung_info.csv").set_index("patient_id")

ids_train, ids_val, ids_test = get_split(0)

BATCH_SIZE = 4
EPOCHS = 100
STEPS_PER_EPOCH = int(510 // BATCH_SIZE)  # 5093 slices
STEPS_PER_EPOCH = 200

ds = get_tf_data(
    h5_file,
    clinical_df,
    patient_list=ids_train,
    local_inpainting=False,
    n_channels=2,
    painting_method="random").repeat().batch(BATCH_SIZE).take(STEPS_PER_EPOCH)

for images in ds.take(1).as_numpy_iterator():
    pass

model_s = UnetLightDecorrelated(output_channels=10, last_activation="linear")


def correlation_loss(y_pred):
    channels = tf.shape(y_pred)[-1]
    y = tf.reshape(y_pred, [-1, channels])
    y -= tf.reduce_mean(y, axis=0)
    corr_mat = tf.matmul(tf.transpose(y), y)
    print(f"corr_mat {corr_mat.numpy()}")
    l = tf.linalg.cholesky(corr_mat)
    return -tf.linalg.trace(tf.math.log(l))


c = correlation_loss(model_s(images[0]))
print("yo")