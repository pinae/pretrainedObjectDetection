# -*- coding: utf-8 -*-
import cv2


class FrameReadError(Exception):
    pass


class WebCam:
    def __enter__(self):
        self.capture_device = cv2.VideoCapture(0)
        if not self.capture_device.isOpened():
            self.capture_device.open()
        return self

    def get_frame(self):
        successfully_read, captured_frame = self.capture_device.read()
        if successfully_read:
            return captured_frame
        else:
            raise FrameReadError("Could not read from webcam: " + str(self.capture_device))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture_device.release()


if __name__ == "__main__":
    with WebCam() as camera:
        while True:
            frame = camera.get_frame()
            cv2.imshow('captured frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
