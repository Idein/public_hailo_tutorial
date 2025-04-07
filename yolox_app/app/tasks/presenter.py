import io
import os
import time

import actfw_core
import consts
import cv2
import numpy as np
from actfw_core.system import get_actcast_firmware_type
from actfw_core.task import Consumer
from actfw_raspberrypi.vc4 import Display


def paste_containCV(canvas, src):
    canvasH, canvasW, _ = canvas.shape
    srcH, srcW, _ = src.shape
    if srcW == canvasW and srcH == canvasH:
        return src
    scale_w = canvasW / srcW
    scale_h = canvasH / srcH
    scale = min(scale_h, scale_w)
    scaledW, scaledH = (int(srcW * scale), int(srcH * scale))
    resized = cv2.resize(src, (scaledW, scaledH))
    offsetW = (canvasW - scaledW) // 2
    offsetH = (canvasH - scaledH) // 2
    canvas[offsetH:offsetH + scaledH, offsetW:offsetW + scaledW] = resized
    return canvas


class WrappedArray:
    # This wrapper have save() method similar to Pillow for the update_image() method.
    def __init__(self, array: np.ndarray):
        self.array = array

    def save(self, bytesio: io.BytesIO, format='PNG'):
        bgr_image = self.array[..., ::-1]
        _, buffer = cv2.imencode(f'.{format.lower()}', bgr_image)
        bytesio.write(buffer)

    def copy(self):
        copied_array = self.array.copy()
        return WrappedArray(copied_array)


class Presenter(Consumer):
    def __init__(self, cmd, use_display):
        super(Presenter, self).__init__()
        in_size = (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT)
        self.cmd = cmd
        if use_display:
            display = Display()
            display_width, display_height = display.size()
            scale = min(
                float(display_width / in_size[0]), float(display_height / in_size[1]))
            width = int(scale * in_size[0])
            height = int(scale * in_size[1])
            left = (display_width - width) // 2
            upper = (display_height - height) // 2
            bg_layer, layer = ((1, 2) if get_actcast_firmware_type() == "raspberrypi-bullseye" else (1000, 2000))
            self.preview_window = display.open_window(
                (left, upper, width, height), in_size, layer)
            self.canvas = np.zeros((in_size[1], in_size[0], 3), dtype=np.uint8)
            self.paste_contain = paste_containCV
        else:
            self.preview_window = None
            self.canvas = None

    def proc(self, inputs):
        cv_img = inputs
        if os.getenv("ACTCAST_ACT_ID"):
            actfw_core.heartbeat()
        self.cmd.update_image(WrappedArray(cv_img))  # update Take Photo image
        if self.preview_window is not None:
            image = self.paste_contain(self.canvas, cv_img)
            self.preview_window.blit(image.tobytes())
            self.preview_window.update()
