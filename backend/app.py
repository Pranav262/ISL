"""
SignBridge — Flask Backend API (MobileNetV2 Engine)
Uses MobileNetV2 CNN (87.3% accuracy) for sign prediction and MediaPipe
in parallel for visual hand skeleton overlay.
"""

import os
import base64
import logging
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model", "best_model.keras")
LABEL_PATH   = os.path.join(BASE_DIR, "model", "label_classes.npy")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# ── Load Model ────────────────────────────────────────────────────────────────
log.info("Loading MobileNetV2 model from %s …", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
label_classes = np.load(LABEL_PATH, allow_pickle=True)
log.info("MobileNetV2 model loaded — %d classes: %s", len(label_classes), list(label_classes))

# ── MediaPipe (for Visual Skeleton Overlay only) ──────────────────────────────
mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.4,
)

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

CONFIDENCE_THRESHOLD = 35.0  # % threshold


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_frame(data_url: str) -> np.ndarray:
    """Decode a Base64 data-URL (image/jpeg) -> BGR numpy array."""
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def extract_skeleton_landmarks(img_rgb: np.ndarray):
    """Extract hand landmark coordinates for canvas rendering."""
    res = HANDS.process(img_rgb)
    if not res.multi_hand_landmarks:
        return []
    lm_list = []
    for hand_lm in res.multi_hand_landmarks:
        lm_list.append([[p.x, p.y, p.z] for p in hand_lm.landmark])
    return lm_list


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_type": "MobileNetV2 CNN (87.3% accuracy)",
        "model": MODEL_PATH,
        "classes": list(label_classes),
        "num_classes": int(len(label_classes)),
    })


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    if not body or "frame" not in body:
        return jsonify({"error": "Missing 'frame' field"}), 400

    try:
        frame_bgr = decode_frame(body["frame"])
    except Exception as exc:
        return jsonify({"error": f"Failed to decode frame: {exc}"}), 400

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. Visual hand skeleton extraction for canvas overlay
    landmarks = extract_skeleton_landmarks(frame_rgb)

    # 2. MobileNetV2 image preprocessing
    resized = cv2.resize(frame_rgb, (224, 224))
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    preprocessed = preprocess_input(img_array)

    # 3. Model Prediction
    preds = model.predict(preprocessed, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx]) * 100
    raw = str(label_classes[idx])

    # Top-5 confidence breakdown for UI bar chart
    top5_idx = np.argsort(preds)[::-1][:5]
    all_probs = [
        {"label": str(label_classes[i]), "prob": round(float(preds[i]) * 100, 1)}
        for i in top5_idx
    ]

    letter = (
        "SPACE" if raw == "{"
        else raw.upper() if conf >= CONFIDENCE_THRESHOLD
        else "?"
    )

    return jsonify({
        "detected": True,
        "letter": letter,
        "raw_label": raw,
        "confidence": round(conf, 1),
        "all_probs": all_probs,
        "landmarks": landmarks,
    })


@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({"classes": [str(c) for c in label_classes]})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("SignBridge MobileNetV2 backend running -> http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
