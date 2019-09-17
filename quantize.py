# -*- coding: utf-8 -*-
import tensorflow as tf
from model import get_model
from ilsvrc2012_dataset import get_dataset as ilsvrc2012
from ilsvrc2012_dataset import get_representative_data


def convert_saved_model(saved_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = ilsvrc2012()
    return converter.convert()


def convert_keras_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(input_gen=get_representative_data)
    return converter.convert()


if __name__ == "__main__":
    quantized_model = convert_keras_model(get_model())
    print(type(quantized_model))
