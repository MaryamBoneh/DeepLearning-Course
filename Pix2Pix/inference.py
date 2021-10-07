
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path



parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()

model = load_model('generator.h5')

def generate_images(test_input):
    print(test_input)
    image = cv2.imread(test_input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype('float32')

    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    # display_list = [test_input[0], prediction[0]]
    # title = ['Input Image', 'Predicted Image']
    filename = Path(test_input).stem
    cv2.imwrite(f'Pix2Pix/output/{filename}.jpg', prediction)

    # for i in range(2):
    #     plt.subplot(1, 2, i+1)
    #     plt.title(title[i])
    #     # Getting the pixel values in the [0, 1] range to plot.
    #     plt.imshow(display_list[i] * 0.5 + 0.5)
    #     plt.axis('off')
    # plt.show()

generate_images(args.input)