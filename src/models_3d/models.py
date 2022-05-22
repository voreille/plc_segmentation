import tensorflow as tf

from src.models.layers import ResidualLayer3D


def get_pretrained_classifier(path=None, encoder_trainable=False):
    encoder = get_pretrained_encoder(path)
    encoder.trainable = encoder_trainable
    return tf.keras.Sequential([
        encoder,
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def get_pretrained_encoder(path=None):
    model = Unet(output_channels=2, last_activation="linear")
    model.build(input_shape=(None, None, None, None, 2))
    if path:
        pretrained_model = tf.keras.models.load_model(path)
        model.set_weights(pretrained_model.get_weights())
    input_tensor = tf.keras.Input(shape=(None, None, None, 2))
    x = input_tensor
    for block in model.down_stack:
        x = block(x)

    return tf.keras.Model(inputs=input_tensor, outputs=x)


class Unet(tf.keras.Model):

    def __init__(
        self,
        output_channels=1,
        last_activation="sigmoid",
    ):
        super().__init__()
        self.output_channels = output_channels
        self.last_activation = last_activation
        self.down_stack = [
            self.get_first_block(12),
            self.get_down_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
        ]

        self.up_stack = [
            UpBlockLight(96),
            UpBlockLight(48),
            UpBlockLight(24),
            UpBlockLight(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(12, 3, activation='linear', padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(output_channels,
                                   1,
                                   activation=last_activation,
                                   padding='SAME'),
        ])

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer3D(filters, 7, padding='SAME', activation="relu"),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
            ResidualLayer3D(filters, 3, padding='SAME', activation="relu"),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []
        for block in self.down_stack:
            x = block(x, training=training)
            skips.append(x)

        skips = reversed(skips[:-1])

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)

        return self.last(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_channels": self.output_channels,
            "last_activation": self.last_activation,
        })
        return config


class UnetRadiomics(tf.keras.Model):

    def __init__(self,
                 *args,
                 output_channels=1,
                 last_activation="sigmoid",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.down_stack = [
            self.get_first_block(12),
            self.get_down_block(24),
            self.get_down_block(48),
            self.get_down_block(96),
            self.get_down_block(192),
        ]

        self.up_stack = [
            UpBlockLight(96),
            UpBlockLight(48),
            UpBlockLight(24),
            UpBlockLight(24, n_conv=1),
        ]
        self.last = tf.keras.Sequential([
            tf.keras.layers.Conv3D(12, 3, activation='linear', padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(output_channels, 1, padding='SAME'),
            tf.keras.layers.Activation(last_activation, dtype='float32'),
        ], )
        self.radiomics = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation("sigmoid", dtype='float32'),
        ], )

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
            x = block(x, training=training)
            skips.append(x)
        x_middle = x
        skips = reversed(skips[:-1])

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip), training=training)

        return {
            "output_seg": self.last(x, training=training),
            "output_plc": self.radiomics(x_middle, training=training)
        }


class UpBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        upsampling_factor=1,
        filters_output=24,
        n_conv=2,
    ):
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


class UpBlockLight(tf.keras.layers.Layer):

    def __init__(
        self,
        filters,
        n_conv=2,
    ):
        super().__init__()
        self.conv = tf.keras.Sequential()
        for _ in range(n_conv):
            self.conv.add(tf.keras.layers.Conv3D(filters, 3, padding='SAME'))
            self.conv.add(tf.keras.layers.BatchNormalization())
            self.conv.add(tf.keras.layers.ReLU())
        self.trans_conv = tf.keras.Sequential([
            tf.keras.layers.Conv3DTranspose(filters,
                                            2,
                                            strides=(2, 2, 2),
                                            padding='SAME',
                                            activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.concat = tf.keras.layers.Concatenate()

    def call(self, inputs, training=None):
        x, skip = inputs
        x = self.trans_conv(x, training=training)
        x = self.concat([x, skip])
        x = self.conv(x, training=training)
        return x