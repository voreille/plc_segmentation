from pydoc import describe
import random

import tensorflow as tf
import numpy as np
from numpy.random import randint
import tensorflow_addons as tfa
from scipy.ndimage import gaussian_filter

from src.data.data_augmentation import random_rotate


class LocalShuffling(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self,
        density=0.6,
        max_size=50,
        min_size=10,
    ):
        self.density = density
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def call(self, img):

        img = img.numpy()
        n_mask = int((img.shape[0] * img.shape[1]) /
                     ((self.max_size + self.min_size) / 2)**2 * self.density)
        for _ in range(n_mask):
            mask_size = (random.randint(self.min_size, self.max_size),
                         random.randint(self.min_size, self.max_size))
            mask_position = (
                random.randint(0, img.shape[0] - mask_size[0] - 1),
                random.randint(0, img.shape[1] - mask_size[1] - 1),
            )
            window = img[mask_position[0]:mask_position[0] + mask_size[0],
                         mask_position[1]:mask_position[1] + mask_size[1], :]

            nrows, ncols, nchannels = window.shape
            window = np.reshape(window, (nrows * ncols, nchannels))
            np.random.shuffle(window)
            window = np.reshape(window, (nrows, ncols, nchannels))
            img[mask_position[0]:mask_position[0] + mask_size[0],
                mask_position[1]:mask_position[1] + mask_size[1], :] = window
        return img


class HardInPainting(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self,
        density=0.6,
        max_size=50,
        min_size=10,
    ):
        self.density = density
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def call(self, img):
        img = img.numpy()
        n_mask = int((img.shape[0] * img.shape[1]) /
                     ((self.max_size + self.min_size) / 2)**2 * self.density)

        for _ in range(n_mask):
            mask_size = (random.randint(self.min_size, self.max_size),
                         random.randint(self.min_size, self.max_size))
            mask_position = (
                random.randint(0, img.shape[0] - mask_size[0] - 1),
                random.randint(0, img.shape[1] - mask_size[1] - 1),
            )

            img[mask_position[0]:mask_position[0] + mask_size[0],
                mask_position[1]:mask_position[1] + mask_size[1], :] = -1

        return img


class InPainting(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self,
        density=0.6,
        max_size=50,
        min_size=10,
        local_value=True,
    ):
        self.density = density
        self.max_size = max_size
        self.min_size = min_size
        self.local_value = local_value

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def call(self, img):
        img = img.numpy()
        n_mask = int((img.shape[0] * img.shape[1]) /
                     ((self.max_size + self.min_size) / 2)**2 * self.density)
        if not self.local_value:
            bmax = np.max(img, axis=(0, 1))
            bmin = np.min(img, axis=(0, 1))
        for _ in range(n_mask):
            mask_size = (random.randint(self.min_size, self.max_size),
                         random.randint(self.min_size, self.max_size))
            mask_position = (
                random.randint(0, img.shape[0] - mask_size[0] - 1),
                random.randint(0, img.shape[1] - mask_size[1] - 1),
            )
            window = img[mask_position[0]:mask_position[0] + mask_size[0],
                         mask_position[1]:mask_position[1] + mask_size[1], :]

            nrows, ncols, nchannels = window.shape
            if self.local_value:
                bmax = np.max(window, axis=(0, 1))
                bmin = np.min(window, axis=(0, 1))
            for c in range(nchannels):
                window[:, :, c] = np.random.uniform(low=bmin[c],
                                                    high=bmax[c],
                                                    size=(nrows, ncols))
            img[mask_position[0]:mask_position[0] + mask_size[0],
                mask_position[1]:mask_position[1] + mask_size[1], :] = window
        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, s1_min=0.1, s1_max=2., s2_min=0.5, s2_max=4.):
        self.prob = p
        self.s1_min = s1_min
        self.s1_max = s1_max
        self.s2_min = s2_min
        self.s2_max = s2_max

    def __call__(self, image):
        image_shape = image.shape
        image = tf.py_function(self.call, [image], tf.float32)
        image.set_shape(image_shape)
        return image

    def call_on_np(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img[:, :, 0] = gaussian_filter(
            img[:, :, 0],
            sigma=random.uniform(self.s1_min, self.s1_max),
        )
        img[:, :, 1] = gaussian_filter(
            img[:, :, 1],
            sigma=random.uniform(self.s2_min, self.s2_max),
        )

        return img

    def call(self, img):
        return self.call_on_np(img.numpy())


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
        do_it = random.random() <= self.prob
        if not do_it:
            return preprocess_image(img,
                                    ct_clipping=self.ct_clipping,
                                    pt_clipping=self.pt_clipping)

        return preprocess_image(img,
                                ct_clipping=self.random_ct_clipping(),
                                pt_clipping=self.random_pt_clipping())


def get_tf_data(
    file,
    clinical_df,
    output_shape_image=(256, 256),
    num_parallel_calls=None,
    oversample=False,
    patient_list=None,
    random_angle=None,
    shuffle=False,
    ct_clipping=(-1350, 150),
    pt_clipping=(0, 10),
    n_channels=3,
    n_local_transforms=2,
    local_inpainting=True,
    return_image=False,
    local_shuffling=True,
    painting_method="random",
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

    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

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
            n_channels=n_channels,
            output_shape=output_shape_image,
        )

    def tf_parse_image(patient):
        image, = tf.py_function(f, [patient], [tf.float32])
        image.set_shape(output_shape_image + (n_channels, ))
        return image

    # def augment(image):
    #     if random_angle is not None:
    #         image = random_rotate(image, angle=random_angle)
    #     return image

    out_ds = patient_ds.map(
        lambda p: tf_parse_image(p),
        num_parallel_calls=num_parallel_calls)  # .map(augment)

    def global_transfo1(image):
        image = RandomStandardization(p=0.0,
                                      ct_clipping=ct_clipping,
                                      pt_clipping=pt_clipping)(image)
        image = GaussianBlur(1.0)(image)
        return image

    def global_transfo2(image):
        image = RandomStandardization(p=1.0,
                                      ct_clipping=ct_clipping,
                                      pt_clipping=pt_clipping)(image)
        image = GaussianBlur(0.1)(image)
        return image

    def local_transfo(image):
        image = RandomStandardization(p=0.5,
                                      ct_clipping=ct_clipping,
                                      pt_clipping=pt_clipping)(image)
        if painting_method == "local_shuffling":
            image = LocalShuffling(density=0.5)(image)
        elif painting_method == "random":
            image = InPainting(density=0.5,
                               local_value=local_inpainting)(image)
        elif painting_method == "constant":
            image = HardInPainting(density=0.5)(image)
        else:
            raise ValueError(
                f"the painting method {painting_method} is not implementend")
        image = GaussianBlur(0.1)(image)
        return image

    def preprocessing(image):
        return RandomStandardization(p=0.0,
                                     ct_clipping=ct_clipping,
                                     pt_clipping=pt_clipping)(image)

    if return_image:
        out_ds = out_ds.map(lambda image: (
            preprocessing(image),
            global_transfo1(image),
            global_transfo2(image),
        ) + tuple([local_transfo(image) for _ in range(n_local_transforms)]),
                            num_parallel_calls=num_parallel_calls)
    else:
        out_ds = out_ds.map(lambda image: (
            global_transfo1(image),
            global_transfo2(image),
        ) + tuple([local_transfo(image) for _ in range(n_local_transforms)]),
                            num_parallel_calls=num_parallel_calls)

    return out_ds


@tf.function
def random_rotate(image, angle=10):
    random_angle = tf.random.uniform(minval=-angle, maxval=angle,
                                     shape=(1, )) * np.pi / 180.0
    # It is not disclosed in the doc, but rotate ask for angle in radians
    image = tfa.image.rotate(image,
                             random_angle,
                             interpolation="bilinear",
                             fill_mode="constant",
                             fill_value=0)
    return image


def _parse_image(
        patient,
        file=None,
        n_channels=3,
        output_shape=(256, 256),
):
    patient = patient.numpy().decode("utf-8")
    image = file[patient]["image"][()]
    mask = file[patient]["mask"][()]
    positions = np.where((mask[..., 2] + mask[..., 3]) != 0)
    pos_idx = random.randint(0, len(positions[0]) - 1)
    center = (
        positions[0][pos_idx],
        positions[1][pos_idx],
        positions[2][pos_idx],
    )

    origin = np.array(
        [center[0] - output_shape[0] // 2, center[1] - output_shape[1] // 2])
    origin[origin < 0] = 0

    if origin[0] + output_shape[0] > image.shape[0]:
        origin[0] = image.shape[0] - output_shape[0] - 1

    if origin[1] + output_shape[1] > image.shape[1]:
        origin[1] = image.shape[1] - output_shape[1] - 1

    original_image_shape = image.shape
    image = image[origin[0]:origin[0] + output_shape[0],
                  origin[1]:origin[1] + output_shape[1], center[2], :]

    if n_channels == 3:
        image = np.stack(
            [image[..., 0], image[..., 1],
             np.zeros_like(image[..., 0])],
            axis=-1)

    if image.shape[0] != output_shape[0] or image.shape[0] != output_shape[1]:
        raise RuntimeError(
            f"MEN, the original image shape is {original_image_shape}, the origin is {origin}"
        )
    return image


def clip_standardize(image, clipping=(-np.inf, np.inf)):
    image[image < clipping[0]] = clipping[0]
    image[image > clipping[1]] = clipping[1]
    image = (2 * image - clipping[1] - clipping[0]) / (clipping[1] -
                                                       clipping[0])
    return image


def preprocess_image(image, ct_clipping=(-1350, 250), pt_clipping=None):
    image[..., 0] = clip_standardize(image[..., 0], clipping=ct_clipping)
    if pt_clipping is not None:
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
