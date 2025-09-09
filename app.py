from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # or use lang='en'/'en' depending on card language

@app.route('/ocr', methods=['POST'])
def extract_text():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Light preprocessing
        resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Run OCR
        result = ocr.ocr(resized, cls=True)
        print("OCR Result:", result)

        if not result or not result[0]:
            return jsonify({"text": "No text found."})

        extracted_text = ""
        for line in result[0]:
            if isinstance(line, list) and len(line) >= 2:
                extracted_text += line[1][0] + "\n"

        return jsonify({"text": extracted_text.strip()})
    except Exception as e:
        print("OCR Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
