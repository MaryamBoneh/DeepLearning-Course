import cv2
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

model = load_model('generator.h5', compile=False)

def generate_images(test_input):

    image = cv2.imread(test_input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype('float32')
    image = (image / 127.5) -1
    image = tf.expand_dims(image, axis = 0)
    prediction = model(image, training=True)
    prediction = np.array((prediction[0, :, :, :] +1) * 127.5).astype('uint8')
    filename = Path(test_input).stem
    cv2.imwrite(f'{args.output}/{filename}.jpg', prediction)

generate_images(args.input)
