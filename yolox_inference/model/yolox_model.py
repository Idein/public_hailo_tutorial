import os
from ctypes import cdll

import cv2
import numpy as np

from .nms import nms
from .yolox_layer import YoloXLayer


class LetterBoxDecoder():
    def __init__(self, input_size=None, letterbox_size=(640, 640)):
        self.x_offset = 0  # fixed
        self.y_offset = 0  # fixed
        self.bg_color = (114, 114, 114)
        self.letterbox_size = letterbox_size
        self.letterbox_image = np.full((self.letterbox_size[1], self.letterbox_size[0], 3), self.bg_color, dtype=np.uint8)
        self.input_size = input_size
        if input_size is not None:
            self.update_ratio(input_size)

    def update_ratio(self, input_size):
        self.input_size = input_size
        width, height = input_size
        lw, lh = self.letterbox_size
        self.ratio = min(lw / width, lh / height)
        scaled_width, scaled_height = width * self.ratio, height * self.ratio
        self.scaled_image_size = (int(scaled_width), int(scaled_height))

    def get_letterbox_image(self, cv_img, update_ratio=False):
        if self.input_size is None or update_ratio:
            self.update_ratio((cv_img.shape[1], cv_img.shape[0]))
        resized_image = cv2.resize(cv_img, self.scaled_image_size, interpolation=cv2.INTER_LINEAR)
        self.letterbox_image[:self.scaled_image_size[1], :self.scaled_image_size[0]] = resized_image
        return self.letterbox_image

    def decode_box_letter_to_orig(self, boxes):
        if len(boxes) == 0:
            return np.empty((0, 4))
        return np.array([[
            max(0, box[0] / self.ratio),
            max(0, box[1] / self.ratio),
            min(box[2] / self.ratio, self.input_size[0]),
            min(box[3] / self.ratio, self.input_size[1])]
            for box in boxes])


class YoloX_Tiny:
    def __init__(
        self,
        conf_thresh: float,
        iou_thresh: float,
        input_size=None
    ):

        self.out0 = np.zeros((52, 52, 85), dtype=np.float32)
        self.out1 = np.zeros((26, 26, 85), dtype=np.float32)
        self.out2 = np.zeros((13, 13, 85), dtype=np.float32)
        lib_path = os.path.join(os.path.dirname(__file__), "../yolox_tiny.so")
        self.lib = cdll.LoadLibrary(lib_path)
        self.lib.init()
        self.iou_thresh = iou_thresh
        self.yolox_layer = YoloXLayer(n_class=80, conf_thresh=conf_thresh, in_size=(416, 416), output_sizes=[[52, 52], [26, 26], [13, 13]])
        self.lb_decoder = LetterBoxDecoder(input_size=None, letterbox_size=(416, 416))

    def __del__(self):
        self.lib.destroy()

    def infer(self, image):

        self.lib.infer(
            image.ctypes.data,
            self.out0.ctypes.data,
            self.out1.ctypes.data,
            self.out2.ctypes.data
        )
        return self.out0, self.out1, self.out2

    def preprocess(self, image):
        lb_img = self.lb_decoder.get_letterbox_image(image)
        return lb_img

    def postprocess(self, out):
        out0, out1, out2 = out
        boxes, scores, classes = self.yolox_layer.run([out0, out1, out2])
        boxes, scores, classes = nms(boxes, scores, classes, iou_threshold=self.iou_thresh)
        boxes = self.lb_decoder.decode_box_letter_to_orig(boxes)
        return boxes, scores, classes

    def run(self, image):
        x = self.preprocess(image)
        out = self.infer(x)
        return self.postprocess(out)
