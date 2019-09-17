# -*- coding: utf-8 -*-
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.experimental import export_saved_model
from cv2 import resize, INTER_AREA
import numpy as np


def get_model():
    goliath = MobileNetV2(include_top=True, weights='imagenet')
    return goliath


def prepare_input(image):
    image = resize(image, dsize=(224, 224), interpolation=INTER_AREA)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


if __name__ == "__main__":
    model = get_model()
