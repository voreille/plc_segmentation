from SimpleITK.SimpleITK import OrImageFilter
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


@tf.function
def random_rotate(image, mask, angle=10):
    random_angle = tf.random.uniform(minval=-angle, maxval=angle,
                                     shape=(1, )) * np.pi / 180.0
    # It is not disclosed in the doc, but rotate ask for angle in radians
    image = tfa.image.rotate(image,
                             random_angle,
                             interpolation="bilinear",
                             fill_mode="constant",
                             fill_value=0)
    mask = tfa.image.rotate(mask,
                            random_angle,
                            interpolation="nearest",
                            fill_mode="constant",
                            fill_value=0)
    return image, mask