import argparse

import cv2
from bbox_drawer import Drawer
from coco_classes import coco_classes
from model.yolox_model import YoloX_Tiny


def run(args):
    model = YoloX_Tiny(conf_thresh=args.conf_thresh, iou_thresh=0.45)
    img = cv2.imread(args.input)
    boxes, scores, classes = model.run(img)
    bbox_drawer = Drawer(class_names=coco_classes)
    result_img = bbox_drawer.draw(img, boxes, scores, classes)
    cv2.imwrite("result.jpg", result_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dog.jpg")
    parser.add_argument("--conf_thresh", default=0.5, type=float)
    args = parser.parse_args()
    run(args)
