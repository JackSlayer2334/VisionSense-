import streamlit as st
import requests
from PIL import Image

API = "http://127.0.0.1:8000"

st.title("🔮 VisionSense+")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)

    if st.button("Analyze"):
        res = requests.post(f"{API}/analyze", files={"file": uploaded.getvalue()}).json()

        st.subheader("🟦 Object Detection")
        if res["detections"]["labels"]:
            for l,s in zip(res["detections"]["labels"], res["detections"]["scores"]):
                st.write(f"{l} ({s:.2f})")
        else:
            st.write("No objects detected")

        st.subheader("🟨 OCR")
        st.write(res["text"] or "No text found")

        st.subheader("🟩 AI Description")
        st.write(res["description"])

        if st.button("🔊 Speak Description"):
            requests.post(f"{API}/speak", json={"text": res["description"]})
