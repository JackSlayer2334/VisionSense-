from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

from .detection import YOLO_ONNX
from .ocr import OCR
from .llm import LLMClient
from .tts import TTS

app = FastAPI()

detector = YOLO_ONNX()
ocr = OCR()
llm = LLMClient()
tts = TTS()

@app.get("/")
def home():
    return {"message": "VisionSense+ Running!"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    det = detector.predict(img)
    text = ocr.read_text(img)
    desc = llm.describe(det, text)

    return {
        "detections": det,
        "text": text,
        "description": desc
    }
