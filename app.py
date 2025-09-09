import os  
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR

app = Flask(__name__)
CORS(app)

ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can update this later to use `use_textline_orientation=True`

@app.route('/ocr', methods=['POST'])
def extract_text():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Optional preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Run OCR
        result = ocr.ocr(resized, cls=True)

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
    print(f"✅ Server is running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
