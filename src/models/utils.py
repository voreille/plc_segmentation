import tensorflow as tf
import numpy as np


def predict_volume(image, model, batch_size=16):
    prediction = model.predict(np.transpose(image, (2, 0, 1, 3)),
                               batch_size=batch_size)
    return np.transpose(prediction, (1, 2, 0, 3))
