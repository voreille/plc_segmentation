from pathlib import Path
import os
import datetime
import json

import h5py
import numpy as np
import pandas as pd
import click
import tensorflow as tf

from src.data.tf_data_hdf5 import get_tf_data
from src.models.models import classifier_mobilevnet
from src.models.callbacks import EarlyStopping

DEBUG = False

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"

if DEBUG:
    EPOCHS = 3
else:
    EPOCHS = 400

plot_only_gtvl = False


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--split", type=click.INT, default=0)
@click.option("--gpu-id", type=click.STRING, default="0")
@click.option("--random-angle", type=click.FLOAT, default=10)
@click.option("--random-shift", type=click.INT, default=15)
@click.option("--center-on", type=click.STRING, default="GTVt")
@click.option('--oversample/--no-oversample', default=True)
@click.option('--pretrained/--no-pretrained', default=True)
def main(config, split, gpu_id, random_angle, random_shift, center_on,
         oversample, pretrained):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(project_dir / "data/processed/hdf5_2d/data_selected_slices.hdf5", "r")

    if not pretrained:
        n_channels = 2
    else:
        n_channels = 3

    if oversample:
        steps_per_epoch = 40
    else:
        steps_per_epoch = None

    clinical_df = pd.read_csv(
        project_dir /
        "data/clinical_info_with_lung_info.csv").set_index("patient_id")

    with open(splits_path, "r") as f:
        splits_list = json.load(f)

    ids_train = splits_list[split]["train"]
    ids_val = splits_list[split]["val"]
    #     ids_test = splits_list[split]["test"]

    ds_train = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_train,
        random_slice=True,
        shuffle=False,
        oversample=True,
        random_angle=random_angle,
        random_shift=random_shift,
        center_on=center_on,
        n_channels=n_channels,
    ).map(lambda x, y, plc_status, patient: (x, plc_status)).batch(16)
    ds_val = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_val,
        center_on="GTVt",
        random_slice=False,
        n_channels=n_channels,
    ).map(lambda x, y, plc_status, patient: (x, plc_status)).batch(4)

    model = classifier_mobilevnet()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=[tf.keras.metrics.Accuracy(),
                 tf.keras.metrics.AUC()],
        run_eagerly=False,
    )

    dir_name = ("classifier__" + f"split_{split}__ovrsmpl_{oversample}__" +
                f"con_{center_on}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()

    if not DEBUG:
        log_dir = str((project_dir / ("logs/fit/" + dir_name)).resolve())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

        callbacks.extend([
            EarlyStopping(
                minimal_num_of_epochs=5,
                monitor='val_auc',
                patience=10,
                verbose=0,
                mode='max',
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