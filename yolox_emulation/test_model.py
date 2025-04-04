import argparse
import cv2
import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import InferenceContext
import tensorflow as tf
from util.yolox_layer import YoloXLayer
from util.nms import nms
from util.bbox_drawer import Drawer

def get_input():
    img = cv2.imread("../../YOLOX/assets/dog.jpg")
    resized_img = cv2.resize(img, (416, 416))
    # input = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) YOLOXは入力はBGR
    input = resized_img.astype(dtype=np.float32)
    input = input[None, :, :, :]
    return resized_img, input

def net_eval(runner, target, images):
    with runner.infer_context(target) as ctx:
        probs_batch = runner.infer(ctx, images)
    return probs_batch

def postprocess(resized_img, out0, out1, out2):
    out0 = out0.reshape(52, 52, 85)
    out1 = out1.reshape(26, 26, 85)
    out2 = out2.reshape(13, 13, 85)
    yolox_layer = YoloXLayer(80, 0.5, in_size=416, output_sizes=[[52, 52], [26, 26], [13, 13]])
    boxes, confs, classes = yolox_layer.run([out0, out1, out2])
    boxes, confs, classes = nms(boxes, confs, classes)
    drawer = Drawer()
    result_img = drawer.draw(resized_img, boxes, confs, classes)
    return result_img

    
def test_quantized():
    quantized_model = "../../yolox_tiny_quantized.har"
    runner = ClientRunner(hw_arch='hailo8', har_path=quantized_model)
    resized_img, input = get_input()
    out0, out1, out2 = net_eval(runner, InferenceContext.SDK_QUANTIZED, input)
    result_img = postprocess(resized_img, out0, out1, out2)
    cv2.imwrite("result_quantized.jpg", result_img)

def test_float():
    model = "../../yolox_tiny.har"
    runner = ClientRunner(hw_arch='hailo8', har_path=model)
    resized_img, input = get_input()
    # input = input / 255. #YOLOXではnormalizationは不要
    out0, out1, out2 = net_eval(runner, InferenceContext.SDK_NATIVE, input)
    result_img = postprocess(resized_img, out0, out1, out2)
    cv2.imwrite("result_float.jpg", result_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    args = parser.parse_args()
    if args.target == "float":
        test_float()
    elif args.target == "quantized":
        test_quantized()