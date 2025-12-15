from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

from .config import INPUT_SIZE, SCORE_THRESHOLD, IOU_THRESHOLD
from .utils import letterbox, xywh2xyxy, scale_boxes, non_max_suppression

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolov8.onnx"

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
    def __init__(self):
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img_lb, ratio, pad = letterbox(img, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, img.shape[2:], ratio, pad

    def postprocess(self, preds, orig_shape, ratio, pad):
        preds = preds[0].transpose(1, 0)   # (8400, 84)

        boxes = preds[:, :4]
        class_logits = preds[:, 4:]
        class_scores = 1 / (1 + np.exp(-class_logits))

        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = scores >= SCORE_THRESHOLD
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        boxes = xywh2xyxy(boxes)
        keep = non_max_suppression(boxes, scores, IOU_THRESHOLD)

        boxes = scale_boxes(boxes[keep], orig_shape, ratio, pad)
        scores = scores[keep]
        labels = [COCO[c] for c in class_ids[keep]]

        return boxes.tolist(), scores.tolist(), labels

    def predict(self, img):
        inp, orig_shape, ratio, pad = self.preprocess(img)
        preds = self.session.run(None, {self.input_name: inp})
        boxes, scores, labels = self.postprocess(preds[0], orig_shape, ratio, pad)
        return {"boxes": boxes, "scores": scores, "labels": labels}

