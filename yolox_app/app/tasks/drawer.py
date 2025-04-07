import time

import consts
import cv2
from actfw_core.task import Pipe


class FPS(object):
    def __init__(self, moving_average=30):
        self.moving_average = moving_average
        self.prev_time = time.time()
        self.dtimes = []

    def update(self):
        cur_time = time.time()
        dtime = cur_time - self.prev_time
        self.prev_time = cur_time
        self.dtimes.append(dtime)
        if len(self.dtimes) > self.moving_average:
            self.dtimes.pop(0)
        return self.get()

    def get(self):
        if len(self.dtimes) == 0:
            return None
        else:
            return len(self.dtimes) / sum(self.dtimes)


class Drawer(Pipe):
    def __init__(self, settings):
        super(Drawer, self).__init__()
        self.fps = FPS(30)

    def proc(self, inputs):
        captured_image, bboxes, scores, classes = inputs
        self.fps.update()
        fps = self.fps.get()
        if fps is None:
            fps_txt = 'FPS: N/A'
        else:
            fps_txt = 'FPS: {:>6.3f}'.format(fps)

        cv2.putText(captured_image, fps_txt, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        if bboxes is not None:
            cols = [(255, 0, 0), (0, 255, 0)]  # 赤、緑
            for box, score, cls in zip(bboxes, scores, classes):
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(captured_image, (x1, y1), (x2, y2), cols[0], 2)
                text = consts.coco_labels[cls] + ":{:.2f}%".format(score * 100)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(captured_image, (x1, y1 - text_height - baseline),
                              (x1 + text_width, y1), (0, 0, 255), -1)
                cv2.putText(captured_image, text, (x1, y1 - baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return captured_image
