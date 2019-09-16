# -*- coding: utf-8 -*-
from model import get_model, prepare_input, decode_predictions
from capture import WebCam
import cv2


def inference(model, frame):
    input_data = prepare_input(frame)
    predictions = goliath.predict(input_data)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    print('Predicted:')
    for i in range(3):
        print("  %d: %s - %.2f" % (i, decoded_predictions[i][1], decoded_predictions[i][2]))


if __name__ == "__main__":
    goliath = get_model()
    with WebCam() as camera:
        while True:
            frame = camera.get_frame()
            inference(goliath, frame)
            cv2.imshow('captured frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
