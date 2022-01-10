from pathlib import Path
import os
import datetime
import json
from random import sample

import h5py
import numpy as np
import pandas as pd
import click
import tensorflow as tf

from src.model_genesis.tf_data_hdf5 import get_tf_data
from src.models.models_3d import Unet
from src.models.callbacks import EarlyStopping
from src.model_genesis.utils import get_tf_degrade_image

DEBUG = False

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"

if DEBUG:
    EPOCHS = 3
else:
    EPOCHS = 400

split = 0


@click.command()
@click.option("--gpu-id", type=click.STRING, default="0")
@click.option("--input-shape", type=click.INT, default=64)
@click.option("--batch-size", type=click.INT, default=4)
def main(gpu_id, input_shape, batch_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
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
    ids_test = splits_list[split]["test"]

    tf_degrade_image = get_tf_degrade_image()
    ds_train = get_tf_data(h5_file,
                           clinical_df,
                           patient_list=ids_train,
                           output_image_shape=3 * (input_shape, ))
    ds_val = get_tf_data(h5_file,
                         clinical_df,
                         patient_list=ids_val,
                         output_image_shape=3 * (input_shape, ))

    ds_train = ds_train.map(lambda x, p: (tf_degrade_image(x), x)).batch(batch_size)
    ds_val = ds_val.map(lambda x, p: (tf_degrade_image(x), x)).cache().batch(batch_size)

    sample_degraded, sample_images = next(ds_val.take(1).as_numpy_iterator())

    model = Unet(output_channels=2, last_activation="linear")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="MSE",
        metrics=["MAE", "MSE"],
        run_eagerly=False,
    )

    dir_name = ("unet_genesis__" + f"ishp_{input_shape}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()

    if not DEBUG:
        log_dir = str(
            (project_dir / ("logs/fit_genesis/" + dir_name)).resolve())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            sample_pred = model.predict(sample_degraded)

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                ims = sample_images[:2, :, :, 32, 0]
                ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
                ims = ims[..., np.newaxis]
                tf.summary.image("Validation images CT", ims, step=epoch)

                ims = sample_pred[:2, :, :, 32, 0]
                ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
                ims = ims[..., np.newaxis]
                tf.summary.image("Reconstruction CT", ims, step=epoch)

                ims = sample_degraded[:2, :, :, 32, 0]
                ims = ims[..., np.newaxis]
                tf.summary.image("Input CT", ims, step=epoch)

                ims = sample_images[:2, :, :, 32, 1]
                ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
                ims = ims[..., np.newaxis]
                tf.summary.image("Validation images PT", ims, step=epoch)

                ims = sample_pred[:2, :, :, 32, 1]
                ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
                ims = ims[..., np.newaxis]
                tf.summary.image("Reconstruction PT", ims, step=epoch)

                ims = sample_degraded[:2, :, :, 32, 1]
                ims = (ims - np.min(ims)) / (np.max(ims) - np.min(ims))
                ims = ims[..., np.newaxis]
                tf.summary.image("Input PT", ims, step=epoch)

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
            EarlyStopping(
                minimal_num_of_epochs=5,
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
    )
    model_dir = project_dir / ("models/" + dir_name)
    model_dir.mkdir()
    model.save(model_dir / "model_weight")


if __name__ == '__main__':
    main()