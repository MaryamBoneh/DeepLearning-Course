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

model = load_model('generator_model.h5', compile=False)

def generate_images(test_input):

    image = tf.io.read_file(test_input)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.expand_dims(image, axis = 0)
    image = (image / 127.5) - 1
    
    prediction = model(image, training=True)
    prediction = np.squeeze(prediction, axis = 0)
    prediction = (prediction) * 0.5 + 0.5

    filename = Path(test_input).stem
    cv2.imwrite(f'{args.output}/{filename}.jpg', prediction)

generate_images(args.input)