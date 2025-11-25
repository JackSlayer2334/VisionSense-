<h1 align="center">🔮 VisionSense+ Real-Time Multimodal AI Assistance for the Visually Impaired</h1> <p align="center"> <img src="https://img.shields.io/badge/AI-Computer Vision-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/LLM-Scene Understanding-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/Tech-FastAPI-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-YOLOv8-orange?style=for-the-badge"> </p>
🚀 Overview

VisionSense+ is a real-time AI system designed to assist visually impaired users using:

🧠 Multimodal AI

YOLOv8 ONNX for object detection

EasyOCR for reading text

LLM (OpenAI or fallback offline) for scene explanation

TTS (Text-to-Speech) for audio guidance

The goal is simple:

Help blind users understand their surroundings through AI-powered audio descriptions.

This is a production-ready ML engineering portfolio project demonstrating:

Computer Vision

LLM integration

Real-time inference

API engineering

End-to-end AI system design

✨ Features
🔍 Real-Time Object Detection

Detects:

People, cars, bikes

Stairs, chairs, obstacles

Traffic lights

Animals & more

📝 OCR Text Reading

Reads:

Signs

Bus numbers

Menus

Documents

📢 Audio Scene Description

AI generates:

A short natural scene summary

Safety warning

Object overview

🌐 FastAPI Backend

Clean, modular API:

/analyze → upload image → returns detections + text + AI description

🎧 Optional TTS Support

Direct audio feedback for blind users.

🏗️ Project Structure
vision-sense/
│── api/
│ ├── main.py # FastAPI app
│ ├── detection.py # YOLOv8 ONNX inference
│ ├── ocr.py # EasyOCR wrapper
│ ├── llm.py # LLM scene explanation
│ ├── tts.py # Text-to-speech
│ ├── utils.py # Preprocessing & NMS
│ └── config.py # Settings
│
├── models/
│ └── yolov8.onnx # (required) ONNX model file
│
├── samples/ # Sample images to test
├── requirements.txt
├── README.md
└── ...

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/vision-sense.git
cd vision-sense

2️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Get YOLOv8 ONNX model
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640

Move the file:

yolov8n.onnx → models/yolov8.onnx

🏃 Run the Server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Open docs:

http://localhost:8000/docs

🧪 Test the API

Place image:

samples/test.jpg

Then test:

curl -X POST "http://localhost:8000/analyze" \
 -F "file=@samples/test.jpg"

Example Output:

{
"detections": {
"boxes": [...],
"scores": [...],
"labels": ["person", "car"]
},
"text": ["Metro Station Entrance"],
"description": "A person is near the entrance. A car is approaching from the right."
}

📸 Screenshots (add when ready)
[ ] object detection output  
[ ] OCR output  
[ ] FastAPI docs screenshot

🧠 How It Works (Architecture)
Camera → Preprocessing → YOLOv8 ONNX → OCR → LLM → Audio/TTS → Blind User

YOLO detects objects

OCR extracts readable text

LLM combines everything into a description

TTS speaks it aloud

📦 Roadmap

Real-time video streaming

Edge-device support (Raspberry Pi)

Offline small LLM (Llama 3.1 3B)

React Native mobile app

Navigation assistance (GPS-based)

🧑‍💻 Skills Demonstrated

This project showcases:

Computer Vision (ONNX Runtime, preprocessing)

Multimodal ML pipelines

Real-time inference optimization

FastAPI backend design

LLM prompt engineering

API architecture

TTS integration

Model deployment workflow

Perfect for:

ML Engineer

Computer Vision Engineer

AI/ML Intern

AI Research Assistant

📝 License

MIT License

❤️ Acknowledgements

Ultralytics YOLO

EasyOCR

OpenAI GPT Models

FastAPI
