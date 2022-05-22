import random

import tensorflow as tf
import numpy as np
from scipy.ndimage import gaussian_filter, rotate


class LocalShuffling2D(object):
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


class HardInPainting2D(object):
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


class InPainting2D(object):
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

    def call(self, img):
        if hasattr(img, "numpy"):
            img = img.numpy()

        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img[..., 0] = gaussian_filter(
            img[..., 0],
            sigma=random.uniform(self.s1_min, self.s1_max),
        )
        img[..., 1] = gaussian_filter(
            img[..., 1],
            sigma=random.uniform(self.s2_min, self.s2_max),
        )

        return img


class RightAngleRotation(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, mirror=True):
        self.prob = p
        self.mirror = mirror

    def __call__(self, *images):
        do_it = random.random() <= self.prob
        if not do_it:
            return images
        angle = np.random.choice([90, 180, 270])
        flip = np.random.randint(2) == 1 and self.mirror
        output_images = []
        for image in images:
            image_shape = image.shape
            image = tf.py_function(self.call, [image, angle, flip], tf.float32)
            image.set_shape(image_shape)
            output_images.append(image)
        return output_images

    def call(self, img, angle, flip):
        if hasattr(img, "numpy"):
            img = img.numpy()

        img = rotate(img, axes=(1, 0), angle=angle)
        if flip:
            img = np.flip(img, axis=1)
        return img


class Standardization(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self,
                 p=0.5,
                 random=False,
                 ct_clipping=(-1350, 150),
                 pt_clipping=(0, 10),
                 ct_ratio=0.1,
                 pt_bdistribution=((-1, 1), (2, 10))):
        if not random:
            p = 0.0
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


def preprocess_image(image, ct_clipping=(-1350, 250), pt_clipping=None):
    image[..., 0] = clip_standardize(image[..., 0], clipping=ct_clipping)
    if pt_clipping is not None:
        image[..., 1] = clip_standardize(image[..., 1], clipping=pt_clipping)
    return image


def clip_standardize(image, clipping=(-np.inf, np.inf)):
    image[image < clipping[0]] = clipping[0]
    image[image > clipping[1]] = clipping[1]
    image = (2 * image - clipping[1] - clipping[0]) / (clipping[1] -
                                                       clipping[0])
    return image
