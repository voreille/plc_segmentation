import tensorflow as tf

from src.models.layers import ResidualLayer3D


class Unet(tf.keras.Model):
    def __init__(
        self,
        output_channels=1,
        last_activation="sigmoid",
    ):
        super().__init__()
        self.down_stack = [
            self.get_first_block(12),
            self.get_down_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
        ]

        self.up_stack = [
            UpBlock(96, upsampling_factor=8),
            UpBlock(48, upsampling_factor=4),
            UpBlock(24, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(12, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters, 7, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
        ])

    def get_encoder(self):
        pass

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
        return self.last(x, training=training)


class UnetRadiomics(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.down_stack = [
            self.get_first_block(12),
            self.get_down_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
        ]

        self.up_stack = [
            UpBlock(96, upsampling_factor=8),
            UpBlock(48, upsampling_factor=4),
            UpBlock(24, upsampling_factor=2),
            UpBlock(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(12, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(1, 1, activation='sigmoid', padding='SAME'),
        ])
        self.radiomics = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters, 7, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME'),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x)
            skips.append(x)
        x_middle = x
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip))
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last(x), self.radiomics(x_middle)


class UpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 *args,
                 upsampling_factor=1,
                 filters_output=24,
                 n_conv=2,
                 **kwargs):
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv3D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv3DTranspose(filters,
                                                          3,
                                                          strides=(2, 2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if upsampling_factor != 1:
            self.upsampling = tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters_output,
                                       1,
                                       padding='SAME',
                                       activation='relu'),
                tf.keras.layers.UpSampling3D(size=(upsampling_factor,
                                                   upsampling_factor,
                                                   upsampling_factor)),
            ])
        else:
            self.upsampling = None

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x, training=training)
        x = self.concat([x, skip])
        x = self.conv(x, training=training)
        if self.upsampling:
            return x, self.upsampling(x)
        else:
            return x