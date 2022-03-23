from pathlib import Path
import os
import datetime
from time import perf_counter
import json
from itertools import product

import h5py
import numpy as np
import pandas as pd
import click
import tensorflow as tf
from tensorflow_addons.optimizers import AdamW

from src.model_dino.tf_data import get_tf_data, RandomStandardization
from src.models.models import UnetLight, UnetLightDecorrelated
from src.data.utils import get_split

DEBUG = False

project_dir = Path(__file__).resolve().parents[2]
splits_path = project_dir / "data/splits.json"

if DEBUG:
    epochs = 1
    n_slices = 10
else:
    epochs = 100
    n_slices = 2000

plot_only_gtvl = False

model_dict = {
    "UnetLight": UnetLight,
    "UnetLightDecorrelated": UnetLightDecorrelated,
}


@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--split", type=click.INT, default=0)
@click.option("--batch-size", type=click.INT, default=4)
@click.option("--output-channels", type=click.INT, default=100)
@click.option("--gpu-id", type=click.STRING, default="3")
@click.option("--model-name", type=click.STRING, default="UnetLight")
@click.option('--oversample/--no-oversample', default=False)
def main(config, split, batch_size, output_channels, gpu_id, model_name,
         oversample):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    steps_per_epoch = n_slices // batch_size
    tau_s = 0.1
    tau_t = 0.04
    momentum_center = 0.9
    momentum_teacher = 0.996
    weight_decay = 0.04
    weight_decay_end = 0.4
    lr = 1e-3
    min_lr = 1e-6
    # optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    h5_file = h5py.File(
        project_dir / "data/processed/hdf5_2d/data_selected_slices.hdf5", "r")

    clinical_df = pd.read_csv(
        project_dir /
        "data/clinical_info_with_lung_info.csv").set_index("patient_id")

    ids_train, ids_val, _ = get_split(0)
    ds_train = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_train,
        oversample=oversample,
        local_inpainting=False,
        n_channels=2,
        painting_method="random",
        return_image=True,
    ).repeat().batch(batch_size).take(steps_per_epoch)

    ds_val = get_tf_data(
        h5_file,
        clinical_df,
        patient_list=ids_val,
        oversample=oversample,
        local_inpainting=False,
        n_channels=2,
        painting_method="random",
    ).cache().batch(batch_size).take(steps_per_epoch)

    model_t = model_dict[model_name](output_channels=output_channels,
                                     last_activation="linear")

    model_r = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x / tau_s)),
        tf.keras.layers.Conv2D(256, 1, activation="relu"),
        tf.keras.layers.Conv2D(2, 1, activation="linear")
    ])

    inputs = tf.keras.Input(shape=(256, 256, 2))
    x_dino = model_dict[model_name](output_channels=output_channels,
                                    last_activation="linear",
                                    name="student_unet")(inputs)

    x_r = model_r(x_dino)
    model_s = tf.keras.Model(inputs=inputs, outputs=[x_dino, x_r])

    dir_name = (f"{model_name}" + f"split_{split}__ovrsmpl_{oversample}__" +
                "with_reconstruction__" +
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    sample_images = next(ds_val.take(1).as_numpy_iterator())
    _, _ = model_s(sample_images[0])
    _ = model_t(sample_images[0])
    callbacks = list()
    if not DEBUG:
        log_dir = str((project_dir / ("logs/fit_dino/" + dir_name)).resolve())
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        file_writer_image = tf.summary.create_file_writer(log_dir + '/images')

        def log_prediction(epoch, logs):
            # Use the model to predict the values from the validation dataset.
            sample_pred, sample_r = model_s.predict(sample_images[2])

            # Log the confusion matrix as an image summary.
            with file_writer_image.as_default():
                tf.summary.image("Predictions", sample_pred, step=epoch)
                tf.summary.image("Inputs", sample_images[2], step=epoch)

        callbacks.extend([
            tf.keras.callbacks.LambdaCallback(on_epoch_end=log_prediction),
        ])

    momentum_schedule = cosine_scheduler(momentum_teacher, 1, epochs,
                                         steps_per_epoch).astype(np.float32)

    # wd_schedule = cosine_scheduler(
    #     weight_decay,
    #     weight_decay_end,
    #     epochs,
    #     steps_per_epoch,
    # )
    # lr_schedule = cosine_scheduler(
    #     lr,  # linear scaling rule
    #     min_lr,
    #     epochs,
    #     steps_per_epoch,
    #     warmup_epochs=10,
    # )
    lr_schedule = np.ones((epochs * steps_per_epoch)) * 1e-3

    model_dir = project_dir / ("models/dino/" + dir_name)
    model_dir.mkdir()

    chkpt_dir = model_dir / "checkpoints"
    center = None
    total_step = 0
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch, ))
        start_time = perf_counter()

        # Iterate over the batches of the dataset.
        for step, im in enumerate(ds_train):
            optimizer.lr = lr_schedule[step]
            # optimizer.weight_decay = wd_schedule[step]
            dino_loss, r_loss, center = train_step(
                im,
                momentum_teacher=momentum_schedule[total_step],
                momentum_center=momentum_center,
                model_s=model_s,
                model_t=model_t,
                tau_s=tau_s,
                tau_t=tau_t,
                optimizer=optimizer,
                center=center,
                weight_reconstruction=100,
            )

            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    f"Training dino loss: {dino_loss} and reconstruction loss: {r_loss} at step {step} and epoch {epoch}"
                )
                print(
                    f"entropy on batch: {entropy(model_s(im[0])[0],tau=tau_s)}"
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
                # print(f"center evolution {center[:10]}")

            total_step += 1

        model_s.save_weights(chkpt_dir / f"model_s/epoch_{epoch}")
        model_s.save_weights(chkpt_dir / f"model_t/epoch_{epoch}")

    print(f"Time taken:  {perf_counter() - start_time}")

    model_s.save(model_dir / "model_s_weight")
    model_t.save(model_dir / "model_t_weight")


@tf.function
def dino_loss(y_s, y_t, *, center, tau_s, tau_t):
    s = tf.nn.log_softmax(y_s / tau_s, axis=-1)
    t = tf.stop_gradient(tf.nn.softmax((y_t - center) / tau_t, axis=-1))
    return -tf.reduce_mean(tf.reduce_sum(t * s, axis=-1))


@tf.function
def entropy(y_pred, tau):
    y_pred = tf.nn.softmax(y_pred / tau, axis=-1)
    log_y_pred = tf.nn.log_softmax(y_pred / tau, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(y_pred * log_y_pred, axis=-1))


@tf.function
def mse_loss(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred)**2)


@tf.function
def train_step(images, *, momentum_teacher, momentum_center, optimizer, center,
               model_s, model_t, tau_s, tau_t, weight_reconstruction):
    original_image = images[0]
    global_x = images[:2]

    y_t = list()
    for x in global_x:
        y_t.append(model_t(x, training=True))  # Forward pass

    if center is None:
        center = tf.reduce_mean(tf.concat(y_t, axis=0), axis=(0, 1, 2))
    with tf.GradientTape() as tape:
        outputs_s = list()
        for x in images:
            outputs_s.append(model_s(x, training=True))  # Forward pass
        y_s, reconstructed_images = list(zip(*outputs_s))

        loss_1 = 0
        n_loss = 0
        for s_ind, t_ind in product(range(len(y_s)), range(len(y_t))):
            if s_ind == t_ind:
                continue
            loss_1 += dino_loss(y_s[s_ind],
                                y_t[t_ind],
                                center=center,
                                tau_s=tau_s,
                                tau_t=tau_t)
            n_loss += 1
        loss_1 = loss_1 / n_loss

        loss_2 = 0
        n_loss = 0
        for y in reconstructed_images:
            loss_2 += mse_loss(original_image, y)
            n_loss += 1
        loss_2 = loss_2 / n_loss
        loss = loss_1 + weight_reconstruction * loss_2

    # Compute gradients
    gradients = tape.gradient(loss, model_s.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1) for g in gradients]

    # Update weights
    optimizer.apply_gradients(zip(gradients, model_s.trainable_variables))

    for i, w in enumerate(model_t.weights):
        model_t.weights[i].assign(momentum_teacher * w +
                                  (1 - momentum_teacher) *
                                  model_s.get_layer("student_unet").weights[i])

    center = momentum_center * center + (1 - momentum_center) * tf.reduce_mean(
        tf.concat(y_t, axis=0), axis=(0, 1, 2))
    return loss_1, loss_2, center


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


if __name__ == '__main__':
    main()