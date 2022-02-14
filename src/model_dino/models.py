from itertools import product

import tensorflow as tf


def train_dino(*, model_s, model_t, ds, epochs):
    for epoch in range(epochs):
        print("\nStart epoch", epoch)

        for step, real_images in enumerate(ds):
            # Train the discriminator & generator on one batch of real images.
            d_loss, g_loss, generated_images = train_step(images)

            # Logging.
            if step % 200 == 0:
                # Print metrics
                print("discriminator loss at step %d: %.2f" % (step, d_loss))
                print("adversarial loss at step %d: %.2f" % (step, g_loss))

                # Save one generated image
                img = tf.keras.preprocessing.image.array_to_img(
                    generated_images[0] * 255.0, scale=False)
                img.save(
                    os.path.join(save_dir,
                                 "generated_img" + str(step) + ".png"))

            # To limit execution time we stop after 10 steps.
            # Remove the lines below to actually train the model!
            if step > 10:
                break


class DinoModel(tf.keras.Model):

    def __init__(self,
                 model_s,
                 model_t,
                 tau_s=0.1,
                 tau_t=0.04,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_s = model_s
        self.model_t = model_t
        self.tau_s = tau_s
        self.tau_t = tau_t

    def dino_loss(self, y_s, y_t, center=0):
        s = tf.nn.softmax(y_s / self.tau_s, axis=-1)
        t = tf.nn.softmax((y_t - center) / self.tau_t, axis=-1)
        return -tf.reduce_mean(tf.reduce_sum((t * tf.math.log(s)), axis=-1))

    def train_step(self, data):
        global_x = data[:2]

        with tf.GradientTape() as tape:
            y_s = list()
            for x in data:
                y_s.append(self.model_s(x, training=True))  # Forward pass
            y_t = list()
            for x in global_x:
                y_t.append(self.model_t(x, training=True))  # Forward pass
            loss = 0
            for ys, yt in product(y_s, y_t):
                loss += self.dino_loss(ys, yt)

        # Compute gradients
        trainable_vars_s = self.model_s.trainable_variables
        gradients = tape.gradient(loss, trainable_vars_s)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars_s))

        return {m.name: m.result() for m in self.metrics}
