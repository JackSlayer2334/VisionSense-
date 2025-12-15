from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2, numpy as np

from .detection import YOLO_ONNX
from .ocr import OCR
from .llm import LLMClient
from .tts import TTS

app = FastAPI()

detector = YOLO_ONNX()
ocr = OCR()
llm = LLMClient()
tts = TTS()

class SpeakRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "VisionSense+ Running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    detections = detector.predict(img)
    text = ocr.read_text(img)
    description = llm.describe(detections, text)

    tts.speak(description)  # 🔊 AUTO SPEAK

    return {
        "detections": detections,
        "text": text,
        "description": description
    }

@app.post("/speak")
def speak(req: SpeakRequest):
    tts.speak(req.text)
    return {"status": "spoken"}
