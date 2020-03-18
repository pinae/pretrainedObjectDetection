# -*- coding: utf-8 -*-
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from cv2 import resize, INTER_AREA
import numpy as np


def get_model():
    goliath = InceptionV3(include_top=True, weights='imagenet')
    return goliath


def prepare_input(image):
    image = resize(image, dsize=(299, 299), interpolation=INTER_AREA)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


if __name__ == "__main__":
    model = get_model()
