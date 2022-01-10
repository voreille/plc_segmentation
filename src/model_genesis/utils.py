import random

import tensorflow as tf

import numpy as np
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t**(n - i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x):
    points = [[0, 0], [random.random(), random.random()],
              [random.random(), random.random()], [1, 1]]

    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(image, prob=0.5):
    x_degraded = np.copy(image)
    x = np.copy(image)
    img_rows, img_cols, img_deps, n_channels = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_shape_x = random.randint(1, img_rows // 10)
        block_shape_y = random.randint(1, img_cols // 10)
        block_shape_z = random.randint(1, img_deps // 10)
        origin_x = random.randint(0, img_rows - block_shape_x)
        origin_y = random.randint(0, img_cols - block_shape_y)
        origin_z = random.randint(0, img_deps - block_shape_z)
        window = x[origin_x:origin_x + block_shape_x,
                   origin_y:origin_y + block_shape_y,
                   origin_z:origin_z + block_shape_z, :]
        window = np.reshape(
            window,
            [block_shape_x * block_shape_y * block_shape_z, n_channels])
        np.random.shuffle(window)
        window = window.reshape(
            (block_shape_x, block_shape_y, block_shape_z, n_channels))
        x_degraded[origin_x:origin_x + block_shape_x,
                   origin_y:origin_y + block_shape_y,
                   origin_z:origin_z + block_shape_z, :] = window

    return x_degraded


def image_in_painting(x):
    img_rows, img_cols, img_deps, n_channels = x.shape
    cnt = 5
    x_inpainted = np.copy(x)
    while cnt > 0 and random.random() < 0.95:
        block_shape_x = random.randint(img_rows // 6, img_rows // 3)
        block_shape_y = random.randint(img_cols // 6, img_cols // 3)
        block_shape_z = random.randint(img_deps // 6, img_deps // 3)
        origin_x = random.randint(3, img_rows - block_shape_x - 3)
        origin_y = random.randint(3, img_cols - block_shape_y - 3)
        origin_z = random.randint(3, img_deps - block_shape_z - 3)
        inpainting_values = np.random.uniform(size=(block_shape_x,
                                                    block_shape_y,
                                                    block_shape_z, n_channels))

        x_inpainted[origin_x:origin_x + block_shape_x,
                    origin_y:origin_y + block_shape_y,
                    origin_z:origin_z + block_shape_z, :] = inpainting_values
        cnt -= 1
    return x_inpainted


def image_out_painting(x):
    img_rows, img_cols, img_deps = x.shape[:-1]
    x_outpainted = np.random.uniform(size=x.shape)
    block_shape_x = img_rows - random.randint(3 * img_rows // 7,
                                              4 * img_rows // 7)
    block_shape_y = img_cols - random.randint(3 * img_cols // 7,
                                              4 * img_cols // 7)
    block_shape_z = img_deps - random.randint(3 * img_deps // 7,
                                              4 * img_deps // 7)
    origin_x = random.randint(3, img_rows - block_shape_x - 3)
    origin_y = random.randint(3, img_cols - block_shape_y - 3)
    origin_z = random.randint(3, img_deps - block_shape_z - 3)
    x_outpainted[origin_x:origin_x + block_shape_x,
                 origin_y:origin_y + block_shape_y, origin_z:origin_z +
                 block_shape_z, :] = x[origin_x:origin_x + block_shape_x,
                                       origin_y:origin_y + block_shape_y,
                                       origin_z:origin_z + block_shape_z, :]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_shape_x = img_rows - random.randint(3 * img_rows // 7,
                                                  4 * img_rows // 7)
        block_shape_y = img_cols - random.randint(3 * img_cols // 7,
                                                  4 * img_cols // 7)
        block_shape_z = img_deps - random.randint(3 * img_deps // 7,
                                                  4 * img_deps // 7)
        origin_x = random.randint(3, img_rows - block_shape_x - 3)
        origin_y = random.randint(3, img_cols - block_shape_y - 3)
        origin_z = random.randint(3, img_deps - block_shape_z - 3)
        x_outpainted[origin_x:origin_x + block_shape_x,
                     origin_y:origin_y + block_shape_y, origin_z:origin_z +
                     block_shape_z, :] = x[origin_x:origin_x + block_shape_x,
                                           origin_y:origin_y + block_shape_y,
                                           origin_z:origin_z +
                                           block_shape_z, :]
        cnt -= 1
    return x_outpainted


def normalize_image_per_channels(image):
    output_image = np.copy(image)
    ranges = list()
    for k in range(image.shape[-1]):
        im = image[..., k]
        min_value = np.min(im)
        max_value = np.max(im)
        output_image[..., k] = (im - min_value) / (max_value - min_value)
        ranges.append((min_value, max_value))
    return output_image, ranges


def rerange_image(image, ranges):
    output_image = np.copy(image)
    for k, (min_value, max_value) in enumerate(ranges):
        im = image[..., k]
        output_image[..., k] = im * (max_value - min_value) + min_value
    return output_image


def degrade_image(
        image,
        local_rate=0.5,
        nonlinear_rate=0.9,
        paint_rate=0.9,
        inpaint_rate=0.1,  # 1 - paint_rate
):

    image, ranges = normalize_image_per_channels(image)
    if random.random() <= local_rate:
        image = local_pixel_shuffling(image)

    # Apply non-Linear transformation with an assigned probability
    if random.random() < nonlinear_rate:
        image = nonlinear_transformation(image)

    # Inpainting & Outpainting
    if random.random() < paint_rate:
        if random.random() < inpaint_rate:
            # Inpainting
            image = image_in_painting(image)
        else:
            # Outpainting
            image = image_out_painting(image)

    return rerange_image(image, ranges)


def get_tf_degrade_image(
        local_rate=0.5,
        nonlinear_rate=0.9,
        paint_rate=0.9,
        inpaint_rate=0.1,  # 1 - paint_rate
):
    def f(image):
        return degrade_image(
            image,
            local_rate=local_rate,
            nonlinear_rate=nonlinear_rate,
            paint_rate=paint_rate,
            inpaint_rate=inpaint_rate,
        )

    def tf_f(image):
        degraded_image = tf.py_function(f, [image], tf.float32)
        degraded_image.set_shape(image.shape)
        return degraded_image

    return tf_f
