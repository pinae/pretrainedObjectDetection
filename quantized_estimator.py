# -*- coding: utf-8 -*-
from model import prepare_input
from estimator import print_predictions
from capture import WebCam
import tensorflow as tf
import cv2


if __name__ == "__main__":
    david = tf.lite.Interpreter(model_path="david.tflite")
    david.allocate_tensors()
    with WebCam() as camera:
        while True:
            frame = camera.get_frame()
            batch = prepare_input(frame)
            david.set_tensor(david.get_input_details()[0]["index"], batch)
            david.invoke()
            predictions = david.get_tensor(
                david.get_output_details()[0]["index"])
            print_predictions(predictions)
            cv2.imshow('captured frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
