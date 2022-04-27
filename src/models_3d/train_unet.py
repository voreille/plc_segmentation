from pathlib import Path
import os
import datetime
import json
from click.core import batch

import h5py
import numpy as np
import pandas as pd
import click
import tensorflow as tf

from src.models_3d.tf_data import get_tf_data
from src.models_3d.models import UnetRadiomics
from src.models.callbacks import EarlyStopping

DEBUG = False

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"
model_path = project_dir / "models/unet_genesis__20220106-151319/model_weight"

if DEBUG:
    EPOCHS = 3
else:
    EPOCHS = 400

plot_only_gtvl = False


@click.command()
@click.option("--split", type=click.INT, default=0)
@click.option("--gpu-id", type=click.STRING, default="0")
@click.option('--oversample/--no-oversample', default=True)
def main(split, gpu_id, oversample):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    batch_size = 4

    h5_file = h5py.File(
        project_dir /
        "data/processed/hdf5_2d_pet_standardized_lung_slices/data.hdf5", "r")

    clinical_df = pd.read_csv(
        project_dir /
        "data/clinical_info_with_lung_info.csv").set_index("patient_id")

    with open(splits_path, "r") as f:
        splits_list = json.load(f)

    ids_train = splits_list[split]["train"]
    ids_val = splits_list[split]["val"]
    #     ids_test = splits_list[split]["test"]

    steps_per_epoch = len(ids_train) // batch_size + 1

    ds_train = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_train,
        oversample=True,
        output_image_shape=(128, 128, 128),
        shuffle=True,
        random_center=True,
        center_on="special",
    ).batch(batch_size)
    ds_val = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_val,
        oversample=False,
        output_image_shape=(128, 128, 128),
        shuffle=False,
        random_center=False,
        center_on="GTVtl",
    ).cache().batch(4)

    model = UnetRadiomics()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=[tf.keras.metrics.Accuracy(),
                 tf.keras.metrics.AUC()],
        run_eagerly=False,
    )

    dir_name = ("unetradiomics__" + f"split_{split}__ovrsmpl_{oversample}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()

    if not DEBUG:
        log_dir = str(
            (project_dir / ("logs/fit_classifier/" + dir_name)).resolve())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    callbacks.extend([
        EarlyStopping(
            minimal_num_of_epochs=25,
            monitor='val_loss',
            patience=10,
            verbose=0,
            mode='min',
            restore_best_weights=True,
        )
    ])

    model.fit(
        x=ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
    )
    model_dir = project_dir / ("models/" + dir_name)
    model_dir.mkdir()
    model.save(model_dir / "model_weight")


if __name__ == '__main__':
    main()