"""
TODO: change how I hande the number of channels
"""
import random

import tensorflow as tf
import numpy as np
from numpy.random import randint

from src.data.data_augmentation import random_rotate


def get_tf_data(
    file,
    clinical_df,
    output_shape_image=(256, 256),
    random_slice=True,
    center_on="GTVt",
    random_position=False,
    num_parallel_calls=None,
    oversample=False,
    patient_list=None,
    random_angle=None,
    shuffle=False,
    return_complete_gtvl=False,
    n_channels=3,
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
            clinical_df=clinical_df,
            file=file,
            random_slice=random_slice,
            random_position=random_position,
            center_on=center_on,
            output_shape_image=output_shape_image,
            return_complete_gtvl=return_complete_gtvl,
            n_channels=n_channels,
        )

    def tf_parse_image(patient):
        image, mask, plc_status = tf.py_function(
            f, [patient], [tf.float32, tf.float32, tf.float32])
        image.set_shape(output_shape_image + (n_channels, ))
        if return_complete_gtvl:
            mask.set_shape(output_shape_image[:2] + (5, ))
        else:
            mask.set_shape(output_shape_image[:2] + (4, ))
        plc_status.set_shape((1, ))
        return image, mask, plc_status

    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE
    out_ds = patient_ds.map(lambda p: (*tf_parse_image(p), p),
                            num_parallel_calls=num_parallel_calls)
    if random_angle:
        out_ds = out_ds.map(
            lambda x, y, plc_status, patient:
            (*random_rotate(x, y, angle=random_angle), plc_status, patient),
            num_parallel_calls=num_parallel_calls)

    # if not return_plc_status:
    #     out_ds = out_ds.map(lambda x, y, plc_status, p: (x, y, p))

    # if not return_patient_name:
    #     out_ds = out_ds.map(lambda x: x[:-2])

    # if not return_mask:
    #     out_ds = out_ds.map(lambda x: (x[0], ) + (x[2:]))

    return out_ds


def _parse_image(
    patient,
    clinical_df=None,
    file=None,
    random_slice=None,
    random_position=False,
    center_on=None,
    output_shape_image=None,
    return_complete_gtvl=None,
    n_channels=3,
):
    patient = patient.numpy().decode("utf-8")
    sick_lung_axis = int(clinical_df.loc[patient, "sick_lung_axis"])
    plc_status = int(clinical_df.loc[patient, "plc_status"])
    image = file[patient]["image"][()]
    mask = file[patient]["mask"][()]
    n_slices = image.shape[2]
    # pet_mean = np.mean(image[..., 1])
    # pet_std = np.std(image[..., 1])

    bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
    center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
    bb_gtvl = get_bb_mask_voxel(mask[..., 1])
    bb_gtvt = get_bb_mask_voxel(mask[..., 0])
    if random_slice:
        # s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
        if center_on == "GTVt":
            s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
        elif center_on == "GTVl":
            s = randint(bb_gtvl[2] + 1, bb_gtvl[5] - 1)
        elif center_on == "nothing":
            s = randint(1, n_slices - 1)
        elif center_on == "special":
            if plc_status == 1:
                s = randint(bb_gtvl[2] + 1, bb_gtvl[5] - 1)
            else:
                s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
        else:
            raise ValueError("Wrong value for center_on argument")
    else:
        if center_on == "GTVt":
            s = (bb_gtvt[5] + bb_gtvt[2]) // 2
        elif center_on == "GTVl":
            s = (bb_gtvl[5] + bb_gtvl[2]) // 2
        elif center_on == "nothing":
            s = n_slices // 2
        elif center_on == "special":
            if plc_status == 1:
                s = (bb_gtvl[5] + bb_gtvl[2]) // 2
            else:
                s = (bb_gtvt[5] + bb_gtvt[2]) // 2
        else:
            raise ValueError("Wrong value for center_on argument")

    if random_position:
        positions = np.where((mask[:, :, s, 2] + mask[:, :, s, 3]) != 0)
        idx = random.randint(0, len(positions[0]) - 1)
        center = np.array([positions[0][idx], positions[1][idx]])
        origin = np.array([
            center[0] - output_shape_image[0] // 2,
            center[1] - output_shape_image[1] // 2
        ])
        origin[origin < 0] = 0

        if origin[0] + output_shape_image[0] - 1 > image.shape[0]:
            origin[0] = image.shape[0] - output_shape_image[0] - 1

        if origin[1] + output_shape_image[1] - 1 > image.shape[1]:
            origin[1] = image.shape[1] - output_shape_image[1] - 1
        center = origin + np.array(output_shape_image[:2]) // 2

    r = [output_shape_image[i] // 2 for i in range(2)]
    mask = mask[center[0] - r[0]:center[0] + r[0],
                center[1] - r[1]:center[1] + r[1], s, :]
    if plc_status == 1:
        gt_gtvl = mask[..., 1]
        mask_loss = (1 - mask[..., sick_lung_axis] + mask[..., 0] +
                     mask[..., 1])
    else:
        gt_gtvl = np.zeros(mask[..., 1].shape)
        mask_loss = np.ones_like(gt_gtvl)
    mask_loss[mask_loss > 0] = 1
    mask_loss[mask_loss <= 0] = 0
    final_mask = np.stack(
        [mask[..., 0], gt_gtvl, mask[..., 2] + mask[..., 3], mask_loss],
        axis=-1,
    )

    if return_complete_gtvl:
        final_mask = np.concatenate(
            [final_mask, mask[..., 1][..., np.newaxis]], axis=-1)
    image = np.squeeze(image[center[0] - r[0]:center[0] + r[0],
                             center[1] - r[1]:center[1] + r[1], s, :])

    if n_channels == 3:
        image = np.stack(
            [image[..., 0], image[..., 1],
             np.zeros_like(image[..., 0])],
            axis=-1)

    return image, final_mask, np.array([plc_status])


def clip_standardize(image, clipping=(-np.inf, np.inf)):
    image[image < clipping[0]] = clipping[0]
    image[image > clipping[1]] = clipping[1]
    image = (2 * image - clipping[1] - clipping[0]) / (clipping[1] -
                                                       clipping[0])
    return image


def preprocess_image(image, ct_clipping=(-1350, 250), pt_clipping=None):
    image[..., 0] = clip_standardize(image[..., 0], clipping=ct_clipping)
    if pt_clipping:
        image[..., 1] = clip_standardize(image[..., 1], clipping=pt_clipping)
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


class RandomStandardization(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self,
                 p=0.5,
                 ct_clipping=(-1350, 150),
                 pt_clipping=(0, 10),
                 ct_ratio=0.1,
                 pt_bdistribution=((-1, 1), (2, 10))):
        self.prob = p
        self.ct_clipping = ct_clipping
        self.pt_clipping = pt_clipping
        self.ct_ratio = ct_ratio
        self.pt_bdistribution = pt_bdistribution

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def random_ct_clipping(self):
        dr = (self.ct_clipping[1] - self.ct_clipping[0]) * self.ct_ratio
        return (
            self.ct_clipping[0] + random.uniform(-dr, dr),
            self.ct_clipping[1] + random.uniform(-dr, dr),
        )

    def random_pt_clipping(self):
        return (
            self.pt_clipping[0] + random.uniform(self.pt_bdistribution[0][0],
                                                 self.pt_bdistribution[0][1]),
            self.pt_clipping[1] + random.uniform(self.pt_bdistribution[1][0],
                                                 self.pt_bdistribution[1][1]),
        )

    def call(self, img):
        if hasattr(img, "numpy"):
            img = img.numpy()

        random_clip = random.random() <= self.prob
        if random_clip:
            return preprocess_image(img,
                                    ct_clipping=self.random_ct_clipping(),
                                    pt_clipping=self.random_pt_clipping())

        return preprocess_image(img,
                                ct_clipping=self.ct_clipping,
                                pt_clipping=self.pt_clipping)
