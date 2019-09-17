# -*- coding: utf-8 -*-
from model import get_model
import tensorflow as tf


if __name__ == "__main__":
    converter = tf.lite.TFLiteConverter.from_keras_model(get_model())
    converted_binary_model = converter.convert()
    with open("goliath.tflite", 'bw') as model_file:
        model_file.write(converted_binary_model)
