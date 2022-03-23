import tensorflow as tf
from tensorflow.python.keras.backend import dropout
from tensorflow.python.training.tracking import base
from tensorflow_addons.losses import sigmoid_focal_crossentropy

from src.models.layers import ResidualLayer2D
from src.decorrelated_bn.normalization import DecorelationNormalization

import matplotlib.pyplot as plt

OUTPUT_CHANNELS = 3


def upsample(
    filters,
    size,
    kind="trans-conv",
):

    result = tf.keras.Sequential()
    if kind == "trans_conv":
        result.add(
            tf.keras.layers.Conv2DTranspose(filters,
                                            2,
                                            strides=2,
                                            padding='same',
                                            activation="relu"))
        result.add(tf.keras.layers.Conv2D(filters, size, padding='same'))
    elif kind == "upsampling":
        result.add(tf.keras.layers.UpSampling2D(interpolation="bilinear"))
        result.add(tf.keras.layers.Conv2D(filters, size, padding='same'))
    elif kind == "old":
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
            ))

    else:
        raise ValueError(f"{kind} is not handled for the kind of upsampling")

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def get_last(filters,
             size,
             upsampling_kind="trans-conv",
             activation="sigmoid"):
    result = tf.keras.Sequential()
    if upsampling_kind == "trans_conv":
        result.add(
            tf.keras.layers.Conv2DTranspose(filters,
                                            2,
                                            strides=2,
                                            padding='same',
                                            activation="relu"))
        result.add(
            tf.keras.layers.Conv2D(filters,
                                   size,
                                   padding='same',
                                   activation=activation))
    elif upsampling_kind == "upsampling":
        result.add(tf.keras.layers.UpSampling2D(interpolation="bilinear"))
        result.add(
            tf.keras.layers.Conv2D(filters,
                                   size,
                                   padding='same',
                                   activation=activation))
    elif upsampling_kind == "old":
        result.add(
            tf.keras.layers.Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
                activation=activation,
            ))
    return result


def downsample(filters, kernel_size, name=""):
    result = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters, kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(padding="same"),
        ],
        name=name,
    )
    return result


def get_decoder(inputs):
    down_stack = [
        tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(16, 5, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(16, 3, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                downsample(64, 3, name="down_1"),
            ],
            name="first_block",
        ),
        downsample(128, 3, name="down_2"),
        downsample(256, 3, name="down_3"),
        downsample(512, 3, name="down_4"),
    ]

    x1 = down_stack[0](inputs)
    x2 = down_stack[1](x1)
    x3 = down_stack[2](x2)
    x4 = down_stack[3](x3)
    return tf.keras.Model(inputs=inputs, outputs=[x1, x2, x3, x4])


def classifier_mobilevnet(n_class=1, input_shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(shape=(None, None, 3))
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)
    base_model.trainable = False

    last = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_class, activation="sigmoid"),
    ])

    inputs = tf.keras.Input(shape=input_shape)
    middle_output = base_model(inputs)
    x = last(middle_output)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unet_model(
    output_channels,
    input_shape=(None, None, 3),
    upsampling_kind="upsampling",
    pretrained=True,
    last_activation="sigmoid",
):
    if pretrained:
        inputs = tf.keras.layers.Input(shape=(None, None, 3))
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        down_stack.trainable = False
    else:
        inputs = tf.keras.layers.Input(shape=(None, None, 2))
        down_stack = get_decoder(inputs)
        down_stack.trainable = True

    up_stack = [
        upsample(512, 3, kind=upsampling_kind),  # 4x4 -> 8x8
        upsample(256, 3, kind=upsampling_kind),  # 8x8 -> 16x16
        upsample(128, 3, kind=upsampling_kind),  # 16x16 -> 32x32
        upsample(64, 3, kind=upsampling_kind),  # 32x32 -> 64x64
    ]

    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = get_last(output_channels,
                    3,
                    upsampling_kind=upsampling_kind,
                    activation=last_activation)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def unetclassif_model(output_channels,
                      input_shape=(None, None, 3),
                      upsampling_kind="upsampling",
                      pretrained=True):
    if pretrained:
        inputs = tf.keras.layers.Input(shape=(None, None, 3))
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                       include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        down_stack.trainable = False
    else:
        inputs = tf.keras.layers.Input(shape=(None, None, 2))
        down_stack = get_decoder(inputs)
        down_stack.trainable = True

    up_stack = [
        upsample(512, 3, kind=upsampling_kind),  # 4x4 -> 8x8
        upsample(256, 3, kind=upsampling_kind),  # 8x8 -> 16x16
        upsample(128, 3, kind=upsampling_kind),  # 16x16 -> 32x32
        upsample(64, 3, kind=upsampling_kind),  # 32x32 -> 64x64
    ]

    x = inputs

    # x_classif = base_model(x)

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Classifier
    classifier = tf.keras.Sequential(name="classifier")
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())
    classifier.add(tf.keras.layers.Dense(128, activation="relu"))
    classifier.add(tf.keras.layers.Dropout(0.5))
    classifier.add(tf.keras.layers.Dense(64, activation="relu"))
    classifier.add(tf.keras.layers.Dropout(0.5))
    classifier.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    x_classif = classifier(x)

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(output_channels,
                                           3,
                                           strides=2,
                                           activation="sigmoid",
                                           padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=[x, x_classif])


def classif_model(output_channels,
                  input_shape=(256, 256, 3),
                  dropout_rate=0.2):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False)

    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    x = base_model(x)

    # Classifier
    classifier = tf.keras.Sequential(name="classifier")
    classifier.add(tf.keras.layers.GlobalAveragePooling2D())
    classifier.add(tf.keras.layers.Dropout(dropout_rate))
    classifier.add(tf.keras.layers.Dense(2, activation="softmax"))

    return tf.keras.Model(inputs=inputs, outputs=classifier(x))


class UpBlock(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 *args,
                 upsampling_factor=1,
                 filters_output=24,
                 n_conv=2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.upsampling_factor = upsampling_factor
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv2D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                          3,
                                                          strides=(2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if upsampling_factor != 1:
            self.upsampling = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters_output,
                                       1,
                                       padding='SAME',
                                       activation='relu'),
                tf.keras.layers.UpSampling2D(size=(upsampling_factor,
                                                   upsampling_factor)),
            ])
        else:
            self.upsampling = None

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        x = self.conv(x)
        if self.upsampling:
            return x, self.upsampling(x)
        else:
            return x


class Unet(tf.keras.Model):

    def __init__(self, *args, output_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
            self.get_down_block(384),
        ]

        self.up_stack = [
            UpBlock(192, upsampling_factor=8),
            UpBlock(96, upsampling_factor=4),
            UpBlock(48, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation='sigmoid',
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x)


class UnetClassif(tf.keras.Model):

    def __init__(self, *args, output_channels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
            self.get_down_block(384),
        ]

        self.up_stack = [
            UpBlock(192, upsampling_factor=8),
            UpBlock(96, upsampling_factor=4),
            UpBlock(48, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation='sigmoid',
                                   padding='SAME'),
        ])
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        x_classif = self.classifier(x, training=training)
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x), x_classif


class UnetLightBase(tf.keras.Model):

    def __init__(self,
                 *args,
                 output_channels=3,
                 last_activation="sigmoid",
                 include_last=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(8),
            self.get_down_block(16),
            self.get_down_block(32),
            self.get_down_block(64),
            self.get_down_block(128),
        ]
        self.up_stack = [
            self.get_up_block(64),
            self.get_up_block(32),
            self.get_up_block(16),
            self.get_up_block(8)
        ]

        if include_last:
            self.last = self.get_last(output_channels,
                                      activation=last_activation)
        else:
            self.last = None

    def get_last(self, output_channels, activation="linear"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation=activation,
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        raise NotImplementedError()

    def get_up_block(self, filters, n_conv=2):
        raise NotImplementedError()

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)

        if self.last:
            return self.last(x, training=training)
        else:
            return x


class UnetLight(UnetLightBase):

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 7, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer2D(filters, 3, strides=2, padding='SAME'),
            ResidualLayer2D(filters, 3, padding='SAME'),
        ])

    def get_up_block(self, filters, n_conv=2):
        return UpBlockLight(filters, n_conv=n_conv)


class UnetLightDecorrelated(UnetLight):

    def get_last(self, output_channels, activation="linear"):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels,
                                   1,
                                   activation="linear",
                                   padding='SAME'),
            DecorelationNormalization(decomposition="iter_norm_wm",
                                      iter_num=5),
            tf.keras.layers.Activation(activation)
        ])


class UpBlockLight(tf.keras.layers.Layer):

    def __init__(self, filters, *args, n_conv=2, up_conv=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = tf.keras.Sequential()
        for _ in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv2D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        if up_conv:
            self.upsample = tf.keras.layers.Conv2DTranspose(filters,
                                                            2,
                                                            strides=(2, 2),
                                                            padding='SAME',
                                                            activation='relu')
        else:
            self.upsample = tf.keras.layers.UpSampling2D(
                interpolation="bilinear")
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.upsample(x, training=training)
        x = self.concat([x, skip])
        return self.conv(x, training=training)