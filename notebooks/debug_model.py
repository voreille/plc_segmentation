import numpy as np

from src.models.models import UnetLight

model = UnetLight(output_channels=100, last_activation="softmax")
model(np.random.uniform(size=(4, 256, 256, 2)))
print("yo")
