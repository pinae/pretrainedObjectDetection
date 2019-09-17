# -*- coding: utf-8 -*-
from model import get_model, prepare_input, decode_predictions, export_saved_model
from capture import WebCam
from time import time
import cv2
import os


def print_predictions(predictions):
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    print('Predicted:')
    for i in range(3):
        print("  %d: %s - %.2f" % (i, decoded_predictions[i][1], decoded_predictions[i][2]))


def inference(model, frame):
    input_data = prepare_input(frame)
    predictions = model.predict(input_data)
    print_predictions(predictions)


if __name__ == "__main__":
    goliath = get_model()
    if not os.path.exists("goliath"):
        export_saved_model(goliath, "goliath")
    with WebCam() as camera:
        while True:
            frame = camera.get_frame()
            start_time = time()
            inference(goliath, frame)
            end_time = time()
            print("Inference time: %fs" % (end_time - start_time))
            cv2.imshow('captured frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
