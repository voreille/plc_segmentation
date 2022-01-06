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
from src.models.evaluation import evaluate_pred_volume

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
@click.option("--upsampling-kind", type=click.STRING, default="upsampling")
@click.option("--split", type=click.INT, default=0)
@click.option("--alpha", type=click.FLOAT, default=0.75)
@click.option("--w-gtvt", type=click.FLOAT, default=0.0)
@click.option("--w-lung", type=click.FLOAT, default=0.0)
@click.option("--gpu-id", type=click.STRING, default="0")
@click.option("--random-angle", type=click.FLOAT, default=None)
@click.option("--random-shift", type=click.INT, default=None)
@click.option("--center-on", type=click.STRING, default="special")
@click.option("--loss-type", type=click.STRING, default="masked")
@click.option('--oversample/--no-oversample', default=False)
@click.option('--pretrained/--no-pretrained', default=True)
def main(config, upsampling_kind, split, alpha, w_gtvt, w_lung, gpu_id,
         random_angle, random_shift, center_on, loss_type, oversample,
         pretrained):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    h5_file = h5py.File(
        project_dir /
        "data/processed/hdf5_2d_pet_standardized_lung_slices/data.hdf5", "r")

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
    ids_test = splits_list[split]["test"]

    ds_train = get_tf_data(h5_file,
                           clinical_df,
                           patient_list=ids_train,
                           shuffle=True,
                           oversample=oversample,
                           random_angle=random_angle,
                           random_shift=random_shift,
                           center_on=center_on,
                           n_channels=n_channels).batch(16)
    ds_val = get_tf_data(h5_file,
                         clinical_df,
                         patient_list=ids_val,
                         center_on="GTVl",
                         random_slice=False,
                         n_channels=n_channels).batch(4)
    ids_val_pos = [p for p in ids_val if clinical_df.loc[p, "plc_status"] == 1]
    ids_val_neg = [p for p in ids_val if clinical_df.loc[p, "plc_status"] == 0]
    ds_sample = get_tf_data(h5_file,
                            clinical_df,
                            patient_list=ids_val_pos[:2] + ids_val_neg[:1],
                            center_on="GTVt",
                            random_slice=False,
                            n_channels=n_channels).batch(3)

    sample_images, sample_seg = next(ds_sample.take(1).as_numpy_iterator())
    sample_seg = np.stack(
        [sample_seg[..., 0], sample_seg[..., 1], sample_seg[..., -1]], axis=-1)

    model = unet_model(3,
                       upsampling_kind=upsampling_kind,
                       pretrained=pretrained)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=CustomLoss(alpha=alpha, w_lung=w_lung, w_gtvt=w_gtvt, loss_type=loss_type),
        run_eagerly=False,
    )

    dir_name = ("unet__" +
                f"prtrnd_{pretrained}__a_{alpha}__wt_{w_gtvt}__wl_{w_lung}"
                f"upsmpl_{upsampling_kind}__" +
                f"split_{split}__ovrsmpl_{oversample}__" + f"con_{center_on}" +
                f"ltyp_{loss_type}__" +
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
            if plot_only_gtvl:
                sample_pred[..., 0] = 0
                sample_pred[..., 2] = 0

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images",
                                 np.stack(
                                     [
                                         sample_images[..., 0],
                                         sample_images[..., 1],
                                         np.zeros_like(sample_images[..., 0]),
                                     ],
                                     axis=-1,
                                 ),
                                 step=epoch)
                tf.summary.image("Predictions", sample_pred, step=epoch)
                tf.summary.image("GTs", sample_seg, step=epoch)

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
        steps_per_epoch=steps_per_epoch,
    )
    model_dir = project_dir / ("models/" + dir_name)
    model_dir.mkdir()
    model.save(model_dir / "model_weight")
    roc_test = evaluate_pred_volume(
        model,
        ids_test,
        h5_file,
        clinical_df,
        n_channels=n_channels,
    )
    roc_val = evaluate_pred_volume(
        model,
        ids_val,
        h5_file,
        clinical_df,
        n_channels=n_channels,
    )
    print(f"The ROC AUC for the val and "
          f"test are {roc_val} and {roc_test} respectively.")


if __name__ == '__main__':
    main()