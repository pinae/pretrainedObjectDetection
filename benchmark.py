# -*- coding: utf-8 -*-
from ilsvrc2012_dataset import get_dataset
from model import preprocess_input, decode_predictions, get_model
from time import time
import tensorflow as tf
import numpy as np
import json


def inference_keras(batch, model):
    return model.predict(batch)


def inference_tflite(batch, model):
    model.set_tensor(model.get_input_details()[0]["index"], batch)
    model.invoke()
    return model.get_tensor(
        model.get_output_details()[0]["index"])


def benchmark(dataset, inference_function, *args):
    count = 0
    correct_predictions = 0
    inference_time_sum = 0.0
    for image, label in dataset:
        start_time = time()
        batch = np.expand_dims(image, axis=0)
        batch = preprocess_input(batch)
        predictions = inference_function(batch, *args)
        end_time = time()
        inference_time_sum += (end_time - start_time)
        decoded_prediction = decode_predictions(predictions, top=1)[0]
        if decoded_prediction[0][0] == ilsverc2012_class_index[str(int(label))]['id']:
            correct_predictions += 1
        count += 1
    return correct_predictions/count, inference_time_sum/count


if __name__ == "__main__":
    with open("ILSVRC2012_labels_map.json", 'r') as class_index_file:
        ilsverc2012_class_index = json.load(class_index_file)
    dataset = get_dataset()
    if False:
        print("Validating Goliath with Keras:")
        goliath = get_model()
        benchmark(dataset.take(5), inference_keras, goliath)  # burn in
        accuracy, avg_inference_time = benchmark(dataset, inference_keras, goliath)
        print("Accuracy: %.2f - Average inference time: %.2f" %
              (accuracy, avg_inference_time))
        print("------------------------------------------")
    print("Validating Goliath with tflite:")
    goliath = tf.lite.Interpreter(model_path="goliath.tflite")
    goliath.allocate_tensors()
    benchmark(dataset.take(5), inference_tflite, goliath)  # burn in
    accuracy, avg_inference_time = benchmark(dataset, inference_tflite, goliath)
    print("Accuracy: %.2f - Average inference time: %.2f" %
          (accuracy, avg_inference_time))
    print("------------------------------------------")
    print("Validating David with tflite:")
    david = tf.lite.Interpreter(model_path="david.tflite")
    david.allocate_tensors()
    benchmark(dataset.take(5), inference_tflite, david)  # burn in
    accuracy, avg_inference_time = benchmark(dataset, inference_tflite, david)
    print("Accuracy: %.2f - Average inference time: %.2f" %
          (accuracy, avg_inference_time))
