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
    random_shift=None,
    num_parallel_calls=None,
    oversample=False,
    patient_list=None,
    random_angle=None,
    shuffle=False,
    return_complete_gtvl=False,
    return_patient_name=False,
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
            clinical_df=clinical_df,
            file=file,
            random_slice=random_slice,
            random_shift=random_shift,
            center_on=center_on,
            output_shape_image=output_shape_image,
            return_complete_gtvl=return_complete_gtvl,
            ct_clipping=ct_clipping,
        )

    def tf_parse_image(patient):
        image, mask = tf.py_function(f, [patient], [tf.float32, tf.float32])
        image.set_shape(output_shape_image + (3, ))
        if return_complete_gtvl:
            mask.set_shape(output_shape_image[:2] + (5, ))
        else:
            mask.set_shape(output_shape_image[:2] + (4, ))
        return image, mask

    out_ds = patient_ds.map(tf_parse_image,
                            num_parallel_calls=num_parallel_calls)
    if random_angle:
        out_ds = out_ds.map(
            lambda x, y: random_rotate(x, y, angle=random_angle),
            num_parallel_calls=num_parallel_calls)
    if return_patient_name:
        out_ds = tf.data.Dataset.zip(
            (out_ds, patient_ds)).map(lambda x, y: (*x, y))

    return out_ds


def _parse_image(
    patient,
    clinical_df=None,
    file=None,
    random_slice=None,
    random_shift=None,
    center_on=None,
    output_shape_image=None,
    return_complete_gtvl=None,
    ct_clipping=[-1350, 150],
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
        if center_on == "GTV T":
            s = randint(bb_gtvt[2] + 1, bb_gtvt[5] - 1)
        elif center_on == "GTV L":
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
        if center_on == "GTV T":
            s = (bb_gtvt[5] + bb_gtvt[2]) // 2
        elif center_on == "GTV L":
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

    if random_shift:
        center += np.array([
            randint(-random_shift, random_shift),
            randint(-random_shift, random_shift)
        ])

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

    image = preprocess_image(
        image,
        ct_clipping=ct_clipping,
        #  pet_mean=pet_mean,
        # pet_std=pet_std,
    )

    return image, final_mask


def preprocess_image(
    image,
    ct_clipping=[-1350, 150],
    # pet_mean=0.0,
    # pet_std=1.0,
):
    ct = image[..., 0]
    ct[ct < ct_clipping[0]] = ct_clipping[0]
    ct[ct > ct_clipping[1]] = ct_clipping[1]
    ct = (2 * ct - ct_clipping[1] - ct_clipping[0]) / (ct_clipping[1] -
                                                       ct_clipping[0])
    # image[..., 1] = (image[..., 1] - pet_mean) / pet_std
    image[..., 0] = ct
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
