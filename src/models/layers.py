import tensorflow as tf


class ResidualLayerBase(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv = None
        self.residual_conv = None
        self.activation = None

    def call(self, x, training=None):
        residual = self.residual_conv(
            x, training=training) if self.residual_conv else x
        return self.activation(self.conv(x, training=training) + residual)


class ResidualLayer2D(ResidualLayerBase):

    def __init__(self, *args, activation='relu', padding="SAME", **kwargs):
        super().__init__(kwargs.get("name"))
        self.filters = args[0]
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(*args,
                                   **kwargs,
                                   padding=padding,
                                   activation="linear"),
            tf.keras.layers.BatchNormalization(),
        ])
        self.activation = tf.keras.layers.Activation(activation)
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[-1]
        if input_shape[-1] != self.filters:
            self.residual_conv = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.filters,
                                       1,
                                       activation=self.activation,
                                       strides=self.strides),
                tf.keras.layers.BatchNormalization()
            ])


class ResidualLayer3D(ResidualLayerBase):

    def __init__(self, *args, activation='relu', padding="SAME", **kwargs):
        super().__init__(kwargs.get("name"))
        self.filters = args[0]
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv3D(*args,
                                   **kwargs,
                                   padding=padding,
                                   activation="linear"),
            tf.keras.layers.BatchNormalization(),
        ])
        self.activation = tf.keras.layers.Activation(activation)
        self.strides = kwargs.get("strides", 1)

    def build(self, input_shape):
        self.c_in = input_shape[-1]
        if input_shape[-1] != self.filters:
            self.residual_conv = tf.keras.Sequential([
                tf.keras.layers.Conv3D(self.filters,
                                       1,
                                       activation=self.activation,
                                       strides=self.strides),
                tf.keras.layers.BatchNormalization()
            ])
