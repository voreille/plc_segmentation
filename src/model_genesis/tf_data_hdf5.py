import tensorflow as tf
import numpy as np
from numpy.random import randint

from src.data.data_augmentation import random_rotate


def get_tf_data(
    file,
    clinical_df,
    output_image_shape=(64, 64, 64),
    num_parallel_calls=None,
    oversample=False,
    patient_list=None,
    shuffle=False,
    ct_clipping=[-1350, 150],
):
    """mask: mask_gtvt, mask_gtvl, mask_lung1, mask_lung2

    Args:
        file ([type]): [description]
        clinical_df ([type]): [description]
        output_shape (tuple, optional): [description]. Defaults to (256, 256).
        random_slice (bool, optional): [description]. Defaults to True.
        random_shift ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if patient_list is None:
        patient_list = list(file.keys())

    if oversample:
        patient_list_neg = [
            p for p in patient_list if clinical_df.loc[p, "plc_status"] == 0
        ]
        patient_list_pos = [
            p for p in patient_list if clinical_df.loc[p, "plc_status"] == 1
        ]

        ds_neg = tf.data.Dataset.from_tensor_slices(patient_list_neg)
        ds_pos = tf.data.Dataset.from_tensor_slices(patient_list_pos)
        if shuffle:
            ds_neg = ds_neg.shuffle(150)
            ds_pos = ds_pos.shuffle(150)
        ds_neg = ds_neg.repeat()
        ds_pos = ds_pos.repeat()

        patient_ds = tf.data.experimental.sample_from_datasets(
            [ds_neg, ds_pos], weights=[0.5, 0.5])
    else:
        patient_ds = tf.data.Dataset.from_tensor_slices(patient_list)
        if shuffle:
            patient_ds = patient_ds.shuffle(150)

    def f(patient):
        return _parse_image(
            patient,
            file=file,
            output_image_shape=output_image_shape,
            ct_clipping=ct_clipping,
        )

    def tf_parse_image(patient):
        image = tf.py_function(f, [patient], tf.float32)
        image.set_shape(output_image_shape + (2, ))
        return image

    out_ds = patient_ds.map(lambda p: (tf_parse_image(p), p),
                            num_parallel_calls=num_parallel_calls)

    return out_ds


def _parse_image(
        patient,
        file=None,
        output_image_shape=(64, 64, 64),
        ct_clipping=[-1350, 150],
):
    patient = patient.numpy().decode("utf-8")
    image = file[patient]["image"][()]
    mask = file[patient]["mask"][()]

    origin = pick_random_origin(output_image_shape,
                                mask[..., 2] + mask[..., 3])

    image = np.squeeze(image[origin[0]:origin[0] + output_image_shape[0],
                             origin[1]:origin[1] + output_image_shape[1],
                             origin[2]:origin[2] + output_image_shape[2], :])

    image = preprocess_image(
        image,
        ct_clipping=ct_clipping,
    )

    return image


def pick_random_origin(output_image_shape, mask):
    image_shape = mask.shape
    positions = np.where(mask)
    random_index = np.random.randint(len(positions[0]))
    origin = np.zeros((3, ), dtype=int)
    for k in range(3):
        center = positions[k][random_index]
        radius = output_image_shape[k] // 2
        origin[k] = center - radius
        if origin[k] < 0:
            origin[k] = 0
        elif origin[k] + output_image_shape[k] > image_shape[k]:
            origin[k] = image_shape[k] - output_image_shape[k]

    return origin


def clip(image, clipping=(-np.inf, np.inf)):
    image[image < clipping[0]] = clipping[0]
    image[image > clipping[1]] = clipping[1]
    image = (2 * image - clipping[1] - clipping[0]) / (clipping[1] -
                                                       clipping[0])
    return image


def preprocess_image(image, ct_clipping=(-1350, 250), pt_clipping=None):
    image[..., 0] = clip(image[..., 0], clipping=ct_clipping)
    if pt_clipping:
        image[..., 0] = clip(image[..., 1], clipping=pt_clipping)
    return image


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, y_max, z_max])
