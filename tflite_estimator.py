# -*- coding: utf-8 -*-
from model import prepare_input
from estimator import print_predictions
from capture import WebCam
from time import time
import tensorflow as tf
import cv2


if __name__ == "__main__":
    goliath = tf.lite.Interpreter(model_path="goliath.tflite")
    goliath.allocate_tensors()
    with WebCam() as camera:
        while True:
            frame = camera.get_frame()
            start_time = time()
            batch = prepare_input(frame)
            goliath.set_tensor(goliath.get_input_details()[0]["index"], batch)
            goliath.invoke()
            predictions = goliath.get_tensor(
                goliath.get_output_details()[0]["index"])
            end_time = time()
            print("Inference time: %fs" % (end_time - start_time))
            print_predictions(predictions)
            cv2.imshow('captured frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
