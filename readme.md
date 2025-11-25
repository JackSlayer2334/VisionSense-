🔮 VisionSense+
AI-Powered Multimodal Assistance for the Visually Impaired

Author: Ayushman Yadav

<p align="center"> <img src="https://img.shields.io/badge/Author-Ayushman%20Yadav-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/AI-Computer%20Vision-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/LLM-Scene%20Understanding-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-YOLOv8-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/API-FastAPI-yellow?style=for-the-badge"> </p>
🧠 Overview

VisionSense+ is an intelligent multimodal AI system designed to help visually impaired users understand their surroundings in real-time. It combines:

YOLOv8 ONNX for object detection

EasyOCR for reading text in the environment

GPT-powered LLM for natural-language scene explanation

TTS (Text-to-Speech) for audio feedback

This project demonstrates end-to-end AI engineering, making it ideal for:

ML Engineer internships

AI/Computer Vision roles

Full-stack ML system building

College major/minor projects

✨ Features
🔍 Object Detection (real-time)

Using YOLOv8 (converted to ONNX), VisionSense+ can detect:

People

Cars, bikes, traffic lights

Animals

Chairs, furniture

Obstacles & more (80 COCO classes)

📝 OCR (EasyOCR)

Reads environmental text:

Signboards

Navigation boards

Shop names

Instructions

Documents

🧠 Scene Analysis (LLM)

An LLM combines detected objects + text and generates a helpful, safe, and human-like explanation.

Example output:

“A person is standing 2 meters ahead. A car is approaching from the right. The signboard reads ‘Metro Station Gate A’.”

🔊 Audio Output (TTS)

Scene explanation is spoken aloud for blind users.

🏗️ Project Structure
VisionSense-/
│── api/
│ ├── main.py # FastAPI app
│ ├── detection.py # YOLOv8 ONNX inference
│ ├── ocr.py # Text detection
│ ├── llm.py # Scene explanation (GPT or fallback)
│ ├── tts.py # Text-to-Speech engine
│ ├── utils.py # Preprocessing, NMS, scaling
│ └── config.py # Settings
│
├── models/
│ └── yolov8.onnx # (ignored in git)
│
├── samples/ # Test images
├── requirements.txt
├── README.md
└── ...

⚙️ Installation
1️⃣ Clone the repo
git clone https://github.com/JackSlayer2334/VisionSense-.git
cd VisionSense-

2️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Download YOLO model
pip install ultralytics
yolo predict model=yolov8n.pt source=None
cp ~/.config/Ultralytics/yolov8n.pt models/

5️⃣ Export to ONNX (compatible opset)
yolo export model=models/yolov8n.pt format=onnx opset=12 imgsz=640
mv yolov8n.onnx models/yolov8.onnx

🚀 Run the Server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Visit API interface:

http://localhost:8000/docs

🧪 Test With an Image
curl -X POST "http://localhost:8000/analyze" \
 -F "file=@samples/test.jpg"

📸 Screenshots / Demo (Add later)

Object Detection Output

OCR Output

Scene Explanation Output

API documentation screenshot

🧩 Roadmap

Real-time video streaming

Mobile app (React Native)

Raspberry Pi support

Offline small-LLM mode

Cloud deployment (Railway/Render/AWS)

🧑‍💻 Skills Demonstrated

This project showcases:

Computer Vision (YOLOv8, ONNX Runtime)

NLP + LLM integration

FastAPI backend engineering

Real-time inference pipeline

Modular ML system design

Git & best practices

Clean architecture and scalability

👨‍💻 Author

Ayushman Yadav
AI/ML Developer | Computer Vision | Backend | Data Structures

📝 License

This project is released under the MIT License.
