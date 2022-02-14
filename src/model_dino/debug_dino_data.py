from pathlib import Path
import os

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.model_dino.tf_data import get_tf_data, RandomStandardization, Inpaint

project_dir = Path(__file__).resolve().parents[2]

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
h5_file = h5py.File(
    project_dir / "data/processed/hdf5_2d/data_selected_slices.hdf5", "r")
clinical_df = pd.read_csv(
    project_dir /
    "data/clinical_info_with_lung_info.csv").set_index("patient_id")

ds = get_tf_data(h5_file, clinical_df)

for images in ds.take(1):
    pass

nrm = RandomStandardization(p=0.0)
inpaint = Inpaint()
images = inpaint(images)
print(images.shape)
