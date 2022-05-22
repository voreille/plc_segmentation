from pathlib import Path
import os
import datetime
import json
from telnetlib import GA
from click.core import batch

import h5py
import numpy as np
import pandas as pd
import click
import tensorflow as tf

from src.models_3d.tf_data import get_tf_data
from src.models_3d.models import UnetRadiomics
from src.models.callbacks import EarlyStopping
from src.models.losses import dice_loss, dice_coe_1_hard
from src.data_augmentation.augmentation import (Standardization, GaussianBlur,
                                                RightAngleRotation)

DEBUG = False

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"
model_path = project_dir / "models/unet_genesis__20220106-151319/model_weight"

if DEBUG:
    EPOCHS = 3
else:
    EPOCHS = 400

plot_only_gtvl = False


def evaluate(model, ds):
    dices = list()
    accs = list()
    for x, y in ds:
        y_pred = model.predict(x)
        dices.append(dsc_gtvt(y["output_seg"], y_pred["output_seg"]))
        accs.append(tf.keras.metrics.BinaryAccuracy()(y["output_plc"],
                                                      y_pred["output_plc"]))
    return np.mean(dices), np.mean(accs)


def dsc_gtvt(y_true, y_pred):
    return dice_coe_1_hard(y_true[..., 0],
                           y_pred[..., 0],
                           spatial_axis=(1, 2, 3))


def dsc_gtvl(y_true, y_pred):
    return dice_coe_1_hard(y_true[..., 1],
                           y_pred[..., 1],
                           spatial_axis=(1, 2, 3))


def get_middle_slice(mask):
    output = list()
    for b in range(mask.shape[0]):
        positions = np.where(mask[b, :, :, :, 1] == 1)
        output.append((np.max(positions[2]) + np.min(positions[2])) // 2)
    return output


def segmentation_loss_channel_wise(y_true, y_pred, channel=0):
    dl = dice_loss(y_true[..., channel],
                   y_pred[..., channel],
                   spatial_axis=(1, 2, 3))
    bc = tf.reduce_mean(tf.keras.losses.binary_focal_crossentropy(
        y_true[..., channel][..., tf.newaxis],
        y_pred[..., channel][..., tf.newaxis],
    ),
                        axis=(1, 2, 3))
    return dl + bc


def segmentation_loss(y_true, y_pred):
    return (segmentation_loss_channel_wise(y_true, y_pred, channel=0) +
            segmentation_loss_channel_wise(y_true, y_pred, channel=1))


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


@click.command()
@click.option("--split", type=click.INT, default=0)
@click.option("--gpu-id", type=click.STRING, default="1")
@click.option('--oversample/--no-oversample', default=True)
@click.option('--mixed-precision/--no-mixed-precision', default=True)
def main(split, gpu_id, oversample, mixed_precision):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    configure_gpu()
    batch_size = 4
    if mixed_precision:
        print("You are using mixed precision training")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    h5_file = h5py.File(project_dir / "data/processed/data.hdf5", "r")

    clinical_df = pd.read_csv(
        project_dir /
        "data/clinical_info_with_lung_info.csv").set_index("patient_id")

    with open(splits_path, "r") as f:
        splits_list = json.load(f)

    ids_train = splits_list[split]["train"]
    ids_val = splits_list[split]["val"]
    ids_test = splits_list[split]["test"]

    if DEBUG:
        steps_per_epoch = 4
    else:
        steps_per_epoch = len(ids_train) // batch_size + 1

    preprocessor = Standardization()
    preprocessor_rand = Standardization(random=True)
    righanglerotater = RightAngleRotation(p=0.5)
    blurrer = GaussianBlur(p=0.5)

    def f_random(im, mask, pstatus, patient_id):
        im, mask = righanglerotater(im, mask)
        im = preprocessor_rand(im)
        im = blurrer(im)
        return im, {"output_seg": mask, "output_plc": pstatus}

    def f(im, mask, pstatus, patient_id):
        return preprocessor(im), {"output_seg": mask, "output_plc": pstatus}

    ds_train = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_train,
        shuffle=True,
        oversample=True,
        center_on="GTVt",
        random_center=True,
        return_gtvl=True,
    ).map(f_random, num_parallel_calls=24).batch(batch_size)

    ds_val = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_val + ids_test,
        shuffle=True,
        oversample=False,
        center_on="GTVt",
        random_center=False,
        return_gtvl=True,
    ).map(f).cache().batch(4)

    ids_val_pos = [p for p in ids_val if clinical_df.loc[p, "plc_status"] == 1]
    ids_val_neg = [p for p in ids_val if clinical_df.loc[p, "plc_status"] == 0]
    ds_sample = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_val_pos[:2] + ids_val_neg[:1],
        shuffle=True,
        oversample=False,
        center_on="GTVt",
        random_center=False,
        return_gtvl=True,
    ).map(f).cache().batch(3)
    samples = next(ds_sample.as_numpy_iterator())
    center_slices = get_middle_slice(samples[1]["output_seg"])
    sample_gts = np.stack(
        [
            samples[1]["output_seg"][b, :, :, s, :]
            for b, s in enumerate(center_slices)
        ],
        axis=0,
    )

    sample_images = np.stack(
        [samples[0][b, :, :, s, 0] for b, s in enumerate(center_slices)],
        axis=0,
    )
    sample_images = sample_images[..., np.newaxis]

    model = UnetRadiomics(output_channels=2)
    model.compile(
        loss={
            "output_seg": segmentation_loss,
            "output_plc": tf.keras.losses.BinaryCrossentropy(),
        },
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics={
            "output_seg": [dsc_gtvt, dsc_gtvl],
            "output_plc": [tf.keras.metrics.BinaryAccuracy()],
        },
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

        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            predictions = model.predict(samples[0])
            seg_pred = np.stack(
                [
                    predictions["output_seg"][b, :, :, s, :]
                    for b, s in enumerate(center_slices)
                ],
                axis=0,
            )
            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Validation images",
                                 sample_images,
                                 step=epoch)
                tf.summary.image("Predictions GTVt",
                                 seg_pred[..., [0]],
                                 step=epoch)
                tf.summary.image("GTs GTVt", sample_gts[..., [0]], step=epoch)
                tf.summary.image("Predictions GTVl",
                                 seg_pred[..., [1]],
                                 step=epoch)
                tf.summary.image("GTs GTVl", sample_gts[..., [1]], step=epoch)

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
            EarlyStopping(
                minimal_num_of_epochs=200,
                monitor='val_loss',
                patience=10,
                verbose=0,
                mode='min',
                restore_best_weights=True,
            )
        ])
    print(
        f"Training on {len(ids_train)} samples and validating on {len(ids_val)} samples"
    )
    model.fit(
        x=ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
    )
    print(f"TO BE SURE: compute metrics to compare")
    val_dice_score, val_accuracy = evaluate(model, ds_val)
    print(f"val_dice_score: {val_dice_score}")
    print(f"val_accuracy: {val_accuracy}")

    train_dice_score, train_accuracy = evaluate(model, ds_train)
    print(f"train_dice_score: {train_dice_score}")
    print(f"train_accuracy: {train_accuracy}")

    if DEBUG:
        return

    model_dir = project_dir / ("models/" + dir_name)
    model_dir.mkdir()
    model.save(model_dir / "model_weight")


if __name__ == '__main__':
    main()