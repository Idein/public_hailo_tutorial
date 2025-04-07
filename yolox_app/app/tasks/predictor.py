import consts
import cv2
import numpy as np
from actfw_core.task import Pipe
from model import YoloX_Tiny


class Preprocessor(Pipe):
    def __init__(self, actual_cap_size):
        super(Preprocessor, self).__init__()
        self.actual_cap_size = actual_cap_size

    def get_capture_image(self, captured_image):
        # actfw_core.capture.Frame -> bytes
        captured_image = captured_image.getvalue()
        # bytes -> np.ndarray
        captured_image = np.frombuffer(captured_image, dtype=np.uint8).reshape(self.actual_cap_size[1], self.actual_cap_size[0], 3)
        h, w, _ = captured_image.shape
        if (w, h) != (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT):
            captured_image = cv2.resize(captured_image, (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT))
        return captured_image

    def proc(self, captured_image):
        captured_image = self.get_capture_image(captured_image)
        return captured_image


class Predictor(Pipe):
    def __init__(self, thresh):
        super(Predictor, self).__init__()
        self.model = YoloX_Tiny(conf_thresh=thresh, iou_thresh=0.45)

    def proc(self, captured_image):
        bboxes, scores, classes = self.model.run(captured_image)
        return captured_image, bboxes, scores, classes
