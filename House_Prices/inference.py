import cv2, os
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("weights/cnn_model.h5")
inputImages = []
outputImage = np.zeros((64, 64, 3), dtype="uint8")

for image_name in os.listdir('images'):
    image = cv2.imread('images/' + image_name)
    image = cv2.resize(image, (32, 32))
    inputImages.append(image)

outputImage[0:32, 0:32] = inputImages[0]
outputImage[0:32, 32:64] = inputImages[1]
outputImage[32:64, 32:64] = inputImages[2]
outputImage[32:64, 0:32] = inputImages[3]
outputImage = outputImage / 255.0
outputImage = np.expand_dims(outputImage, axis= 0)

pred = model.predict([outputImage])[0]
print('price: ', pred)
