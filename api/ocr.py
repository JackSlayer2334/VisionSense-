import easyocr

class OCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_text(self, img):
        results = self.reader.readtext(img)
        return [text for _, text, conf in results if conf > 0.4]
