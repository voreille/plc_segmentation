from random import shuffle

import numpy as np


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, y_max, z_max])


def get_lung_volume(patient, file, output_shape_image=None):
    image = file[patient]["image"][()]
    mask = file[patient]["mask"][()]
    if output_shape_image is not None:
        bb_lung = get_bb_mask_voxel(mask[..., 2] + mask[..., 3])
        center = ((bb_lung[:3] + bb_lung[3:]) // 2)[:2]
        r = [output_shape_image[i] // 2 for i in range(2)]
        mask = mask[center[0] - r[0]:center[0] + r[0],
                    center[1] - r[1]:center[1] + r[1],
                    bb_lung[2]:bb_lung[-1], :]

        image = image[center[0] - r[0]:center[0] + r[0],
                      center[1] - r[1]:center[1] + r[1],
                      bb_lung[2]:bb_lung[-1], :]
    return image, mask


def get_split(patient_list, clinical_info):
    """
    The split are 50, 10, 46 for training, val and test
    in the val and test we have 50 - 50 PLC in the train 75 % 
    """

    patient_list.remove("PatientLC_63")  # Just one lung
    patient_list.remove("PatientLC_72")  # the same as 70

    plc_pos_ids = [
        p for p in patient_list if clinical_info.loc[p, "plc_status"] == 1
    ]

    plc_neg_ids = [
        p for p in patient_list if clinical_info.loc[p, "plc_status"] == 0
    ]
    shuffle(plc_neg_ids)
    shuffle(plc_pos_ids)

    ids_test = plc_neg_ids[:23] + plc_pos_ids[:23]
    ids_val = plc_neg_ids[23:28] + plc_pos_ids[23:28]
    ids_train = plc_neg_ids[28:] + plc_pos_ids[28:]

    return ids_train, ids_val, ids_test
