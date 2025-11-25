# detection.py

import onnxruntime as ort
import numpy as np
import cv2

from .config import *
from .utils import letterbox, xywh2xyxy, scale_boxes, non_max_suppression

# COCO class names
COCO = [
 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
 'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
 'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
 'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
 'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
 'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
 'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
 'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush'
]

class YOLO_ONNX:
    def __init__(self, model_path=MODEL_PATH):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = INPUT_SIZE

    def preprocess(self, img):
        img_letterboxed, ratio, pad = letterbox(img, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_norm, (2,0,1))
        img_input = np.expand_dims(img_transposed, 0)
        return img_input, img.shape[:2], ratio, pad

    def postprocess(self, preds, orig_shape, ratio, pad):
        preds = preds[0]  # shape: (N, 84)
        xywh = preds[:, :4]
        obj_conf = preds[:, 4]
        class_conf = preds[:, 5:]

        class_ids = np.argmax(class_conf, axis=1)
        class_scores = class_conf[np.arange(len(class_ids)), class_ids]
        scores = obj_conf * class_scores

        mask = scores > SCORE_THRESHOLD
        if not np.any(mask):
            return [], [], []

        xywh = xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        boxes = xywh2xyxy(xywh)
        boxes = boxes.astype(np.float32)

        keep = non_max_suppression(boxes, scores, IOU_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        boxes = scale_boxes(boxes, orig_shape, ratio, pad)

        labels = [COCO[c] for c in class_ids]
        return boxes.tolist(), scores.tolist(), labels

    def predict(self, img):
        inp, orig_shape, ratio, pad = self.preprocess(img)
        preds = self.session.run(None, {self.input_name: inp})
        boxes, scores, labels = self.postprocess(np.array(preds[0]), orig_shape, ratio, pad)
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
