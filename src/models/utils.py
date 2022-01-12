import tensorflow as tf
import numpy as np


def predict_volume(image, model, batch_size=16, multitask=False):

    prediction = model.predict(np.transpose(image, (2, 0, 1, 3)),
                               batch_size=batch_size)
    if multitask:
        prediction = prediction[0]

    return np.transpose(prediction, (1, 2, 0, 3))


def reshape_image_unet(image, mask_lung, level=5, p_id=""):
    bb_lung = get_bb_mask_voxel(mask_lung)
    center = ((bb_lung[:3] + bb_lung[3:]) // 2).astype(int)
    lung_shape = np.abs(bb_lung[3:] - bb_lung[:3])
    max_shape = np.max(lung_shape[:2])
    final_shape = max_shape + 2**level - max_shape % 2**level
    radius = int(final_shape // 2)
    image_cropped = image[center[0] - radius:center[0] + radius,
                          center[1] - radius:center[1] + radius, :, :]
    min_shape = np.min(image_cropped.shape[:2])
    if min_shape < final_shape:  # Maybe do some recursion
        final_shape = min_shape - min_shape % 2**level
        print(
            f"THE PATIENT {p_id} has some weird shape going on: {image.shape}")

        radius = int(final_shape // 2)
        image_cropped = image[center[0] - radius:center[0] + radius,
                              center[1] - radius:center[1] + radius, :, :]

    return image_cropped


def get_bb_mask_voxel(mask):
    positions = np.where(mask != 0)
    x_min = np.min(positions[0])
    y_min = np.min(positions[1])
    z_min = np.min(positions[2])
    x_max = np.max(positions[0])
    y_max = np.max(positions[1])
    z_max = np.max(positions[2])
    return np.array([x_min, y_min, z_min, x_max, y_max, z_max])
