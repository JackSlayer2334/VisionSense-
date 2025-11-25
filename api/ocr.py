import easyocr
import cv2
from .config import OCR_LANGS

class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(OCR_LANGS.split(","))

    def read_text(self, image):
        result = self.reader.readtext(image)
        texts = [r[1] for r in result]
        return texts
