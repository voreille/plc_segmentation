import tensorflow as tf
import numpy as np
from numpy.random import randint

from src.data.data_augmentation import random_rotate

masks_dict = {
    "GTVt": lambda m: m[..., 0],
    "GTVl": lambda m: m[..., 1],
    "lung": lambda m: m[..., 2] + m[..., 3],
    "GTVn": lambda m: m[..., 4],
    "GTVtl": lambda m: m[..., 0] + m[..., 1],
    "GTVtln": lambda m: m[..., 0] + m[..., 1] + m[..., 4],
}


def get_tf_data(
    file,
    clinical_df,
    output_image_shape=(128, 128, 128),
    oversample=False,
    patient_list=None,
    shuffle=False,
    ct_clipping=[-1350, 150],
    num_parallel_calls=None,
    return_segmentation=False,
    random_center=False,
    center_on="GTVtl",
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
        if return_segmentation:
            return _parse_image_segmentation(
                patient,
                clinical_df=clinical_df,
                file=file,
                output_image_shape=output_image_shape,
                ct_clipping=ct_clipping,
            )
        else:
            return _parse_image(
                patient,
                clinical_df=clinical_df,
                file=file,
                output_image_shape=output_image_shape,
                ct_clipping=ct_clipping,
                random_center=random_center,
                center_on=center_on,
            )

    def tf_parse_image(patient):
        image, plc_status = tf.py_function(f, [patient],
                                           [tf.float32, tf.float32])
        image.set_shape(output_image_shape + (2, ))
        return image, plc_status

    out_ds = patient_ds.map(lambda p: tf_parse_image(p),
                            num_parallel_calls=num_parallel_calls)

    return out_ds


def _parse_image(
    patient,
    clinical_df=None,
    file=None,
    output_image_shape=None,
    ct_clipping=[-1350, 150],
    random_center=False,
    center_on="GTVl",
):
    patient = patient.numpy().decode("utf-8")
    plc_status = int(clinical_df.loc[patient, "plc_status"])
    image = file[patient]["image"][()]
    mask = file[patient]["mask"][()]
    if center_on == "special":
        if plc_status:
            center_on = "GTVtl"
        else:
            center_on = "GTVtln"

    origin = get_origin(output_image_shape,
                        masks_dict[center_on](mask),
                        random=random_center)

    image = crop_image(image, origin, output_image_shape)

    image = preprocess_image(
        image,
        ct_clipping=ct_clipping,
        #  pet_mean=pet_mean,
        # pet_std=pet_std,
    )

    return image, plc_status


def _parse_image_segmentation(
    patient,
    clinical_df=None,
    file=None,
    output_image_shape=None,
    ct_clipping=[-1350, 150],
):
    raise NotImplementedError("yo c'est pas implémenté")


def get_origin(output_image_shape, mask, random=False):
    image_shape = mask.shape
    if random:
        positions = np.where(mask)
        random_index = np.random.randint(len(positions[0]))
        center = np.array([positions[k][random_index] for k in range(3)])
    else:
        bb = get_bb_mask_voxel(mask)
        center = ((bb[:3] + bb[3:]) / 2).astype(int)
    origin = np.zeros((3, ), dtype=int)
    for k in range(3):
        radius = output_image_shape[k] // 2
        origin[k] = center[k] - radius
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


def crop_image(image, origin, shape):
    return image[origin[0]:origin[0] + shape[0],
                 origin[1]:origin[1] + shape[1],
                 origin[2]:origin[2] + shape[2], :]
