from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import re
from db_connection import get_mongo_conn
from face_model import prepare_model
from scipy.spatial.distance import cosine
import json

app = Flask(__name__)

with open("config.json", "r") as f:
    config = json.load(f)

model = prepare_model(config)
logs_col, basic_col = get_mongo_conn()
COSINE_THRESHOLD = 0.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = data['image']

        # Remove base64 header
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image = base64.b64decode(image_data)
        np_arr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        faces = model.get(frame)
        if not faces:
            return jsonify({"success": False, "message": "No face detected"})


        live_embedding = faces[0].embedding

        for person in basic_col.find():
            db_embedding = np.array(person["person_embedding"])
            similarity = cosine(live_embedding, db_embedding)
            if similarity < COSINE_THRESHOLD:
                return jsonify({
                    "success": True,
                    "person_name": person["person_name"],
                    "person_no": person["person_no"]
                })

        return jsonify({"success": False, "message": "No match found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8094)