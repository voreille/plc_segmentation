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
from src.models.models import unet_model
from src.models.losses import CustomLoss
from src.models.callbacks import EarlyStopping

DEBUG = False

w_gtvl = 105
w_background = 0.5

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"
oversample = True

if DEBUG:
    EPOCHS = 3
else:
    EPOCHS = 400


def loss(y_true, y_pred):
    l = tf.keras.losses.binary_crossentropy(
        tf.expand_dims(y_true[..., 1], axis=-1),
        tf.expand_dims(y_pred[..., 0], axis=-1))
    l *= y_true[..., 3]
    w = y_true[..., 1] * w_gtvl + (1 - y_true[..., 1]) * w_background
    n_elems = tf.reduce_sum(y_true[..., 3], axis=(1, 2))
    return tf.reduce_sum(w * l, axis=(1, 2)) / n_elems


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--upsampling-kind", type=click.STRING, default="upsampling")
@click.option("--split", type=click.INT, default=0)
@click.option("--alpha", type=click.FLOAT, default=0.25)
@click.option("--gpu-id", type=click.STRING, default="0")
@click.option("--random-angle", type=click.FLOAT, default=None)
@click.option("--random-shift", type=click.INT, default=None)
def main(config, upsampling_kind, split, alpha, gpu_id, random_angle,
         random_shift):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(project_dir / "data/processed/hdf5_2d/data.hdf5", "r")

    if oversample:
        steps_per_epoch = 4
    else:
        steps_per_epoch = None

    clinical_df = pd.read_csv(
        project_dir /
        "data/clinical_info_with_lung_info.csv").set_index("patient_id")

    with open(splits_path, "r") as f:
        splits_list = json.load(f)

    ids_train = splits_list[split]["train"]
    ids_val = splits_list[split]["val"]
    ids_test = splits_list[split]["test"]

    ds_train = get_tf_data(h5_file,
                           clinical_df,
                           patient_list=ids_train,
                           shuffle=True,
                           oversample=oversample,
                           random_angle=random_angle,
                           random_shift=random_shift,
                           center_on="special").batch(16)
    ds_val = get_tf_data(h5_file,
                         clinical_df,
                         patient_list=ids_val,
                         center_on="GTV L",
                         random_slice=False).batch(4)
    # ds_test = get_tf_data(h5_file,
    #                       clinical_df,
    #                       patient_list=ids_test,
    #                       center_on="GTV L",
    #                       random_slice=False).batch(4)

    sample_images, sample_seg = next(ds_val.take(1).as_numpy_iterator())
    sample_seg = np.stack(
        [sample_seg[..., 0], sample_seg[..., 1], sample_seg[..., -1]], axis=-1)

    model = unet_model(3, upsampling_kind=upsampling_kind)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=CustomLoss(alpha=alpha, w_lung=0, w_gtvt=0),
    )

    dir_name = ("pretrained_unet__" +
                f"alpha_{alpha}__upsampling_{upsampling_kind}__" +
                f"split_{split}__oversample_{oversample}__" +
                f"rangle_{random_angle}__rshift_{random_shift}__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = list()

    if not DEBUG:
        log_dir = str((project_dir / ("logs/fit/" + dir_name)).resolve())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            sample_pred = model.predict(sample_images)

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images",
                                 sample_images,
                                 step=epoch)
                tf.summary.image("Predictions", sample_pred, step=epoch)
                tf.summary.image("GTs", sample_seg, step=epoch)

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
            EarlyStopping(
                minimal_num_of_epochs=50,
                monitor='val_loss',
                patience=30,
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