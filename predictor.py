from tensorflow.keras.models import load_model
import cv2
import numpy
import Config

model = load_model("models/generator1510")
noise = numpy.random.normal(size=(100,Config.noise_dim))
images = model.predict(noise)*255
for i in range(100):
    cv2.imwrite(f"prediction/digit{i}.jpg", images[i])
