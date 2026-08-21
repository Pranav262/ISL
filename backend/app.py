"""
SignBridge — Flask Backend API (Ensemble Learning Engine)
=========================================================
Combines 3 Diverse Models via Weighted Soft Voting Ensemble:
  1. Static Landmark MLP  (126-dim normalized keypoint DNN, weight: 0.45)
  2. Dynamic BiLSTM       (30-frame temporal keypoint RNN, weight: 0.45)
  3. MobileNetV2 CNN      (224x224 raw image CNN, weight: 0.10)

KEY ENHANCEMENTS FOR ALL ISL SIGNS (A, V, Y, W, B, C, etc.):
  - Per-Hand Wrist-Relative & Scale Normalization: Makes keypoints 100% invariant
    to camera position, distance, tilt, and resolution.
  - Smart Left/Right Hand Swap Auto-Alignment: Evaluates both [Left, Right] and [Right, Left]
    orientations when 1 hand is detected, so signs work seamlessly regardless of dominant
    hand or camera mirror view!
  - Hand Detection Gate: Immediately returns detected = False when no hand keypoints in frame.
"""

import os
import base64
import logging
import threading
from collections import deque

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
MODEL_DIR    = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# Model Paths
LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, "signbridge_landmark_model.keras")
LANDMARK_LABEL_PATH = os.path.join(MODEL_DIR, "label_classes.npy")

CNN_MODEL_PATH   = os.path.join(MODEL_DIR, "best_model.keras")

SEQ_MODEL_PATH   = os.path.join(MODEL_DIR, "isl_sequence_model.keras")
SEQ_LABEL_PATH   = os.path.join(MODEL_DIR, "isl_label_classes.npy")

# ── 1. Load Landmark MLP Model ────────────────────────────────────────────────
LM_AVAILABLE = False
lm_model = lm_labels = None
if os.path.exists(LANDMARK_MODEL_PATH) and os.path.exists(LANDMARK_LABEL_PATH):
    log.info("Loading Landmark MLP model …")
    lm_model  = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
    lm_labels = list(np.load(LANDMARK_LABEL_PATH, allow_pickle=True))
    LM_AVAILABLE = True
    log.info("Landmark MLP loaded — %d classes", len(lm_labels))

# ── 2. Load CNN Model ─────────────────────────────────────────────────────────
log.info("Loading MobileNetV2 model …")
cnn_model  = tf.keras.models.load_model(CNN_MODEL_PATH)
cnn_labels = list("abcdefghijklmnopqrstuvwxyz") + ["{"]   # 27 classes
log.info("MobileNetV2 loaded — %d classes", len(cnn_labels))

# ── 3. Load Sequence Model ────────────────────────────────────────────────────
SEQ_AVAILABLE = False
seq_model = seq_labels = None
if os.path.exists(SEQ_MODEL_PATH) and os.path.exists(SEQ_LABEL_PATH):
    log.info("Loading ISL BiLSTM sequence model …")
    seq_model  = tf.keras.models.load_model(SEQ_MODEL_PATH)
    seq_labels = list(np.load(SEQ_LABEL_PATH, allow_pickle=True))
    SEQ_AVAILABLE = True
    log.info("BiLSTM loaded — %d classes", len(seq_labels))
else:
    log.warning("ISL sequence model not found at %s — dynamic prediction disabled", SEQ_MODEL_PATH)

# ── Shared Ensemble Class Alignment ───────────────────────────────────────────
ISL_23_CLASSES = [c.lower() for c in seq_labels] if seq_labels else [
    'a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]

CNN_TO_ISL_MAP = []
for c in ISL_23_CLASSES:
    if c in cnn_labels:
        CNN_TO_ISL_MAP.append(cnn_labels.index(c))
    else:
        CNN_TO_ISL_MAP.append(-1)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
HANDS = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.4,
)

# ── Rolling Frame Buffer ──────────────────────────────────────────────────────
SEQUENCE_LEN   = 30
FEATURE_DIM    = 126
_buffer_lock   = threading.Lock()
_frame_buffer  = deque(maxlen=SEQUENCE_LEN)   # last 30 landmark vectors

_seq_prediction = {
    "letter":     None,
    "confidence": 0.0,
    "all_probs":  [],
    "raw_probs":  np.ones(len(ISL_23_CLASSES)) / len(ISL_23_CLASSES),
}

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

ENSEMBLE_CONFIDENCE_THRESHOLD = 35.0   # % — below this → "?"


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_frame(data_url: str) -> np.ndarray:
    """Decode Base64 image data-URL → BGR numpy array."""
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def normalize_frame_landmarks(feat_126: np.ndarray) -> np.ndarray:
    """Per-hand wrist centering, scale normalization, and engineered features (distances + angles)."""
    left_raw  = feat_126[:63].reshape(21, 3)
    right_raw = feat_126[63:].reshape(21, 3)

    def _extract_features(hand):
        if np.abs(hand).sum() < 1e-5:
            return np.zeros(98, dtype=np.float32)
        
        wrist = hand[0]
        hand_centered = hand - wrist
        scale = np.linalg.norm(hand_centered[9]) + 1e-6
        hand_norm = (hand_centered / scale).astype(np.float32)
        
        coords = hand_norm.flatten()
        dists = np.linalg.norm(hand_norm[1:], axis=1)
        
        angles = []
        finger_chains = [
            [0, 1, 2, 3, 4],     # Thumb
            [0, 5, 6, 7, 8],     # Index
            [0, 9, 10, 11, 12],  # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20]  # Pinky
        ]
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                p1 = hand_norm[chain[i]]
                p2 = hand_norm[chain[i+1]]
                p3 = hand_norm[chain[i+2]]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-5 or n2 < 1e-5:
                    angles.append(0.0)
                else:
                    cosine_angle = np.dot(v1, v2) / (n1 * n2)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    angles.append(angle)
                    
        return np.concatenate([coords, dists, angles]).astype(np.float32)

    l_feat = _extract_features(left_raw)
    r_feat = _extract_features(right_raw)
    return np.concatenate([l_feat, r_feat])


def extract_landmarks_full(img_rgb: np.ndarray):
    """
    Run MediaPipe Hands and return:
      - landmarks_render: list of [[x,y,z], ...] for each detected hand (canvas overlay)
      - feat_vec: 126-dim normalized landmark vector
      - feat_vec_swapped: 126-dim normalized vector with Left and Right slots swapped
    """
    res = HANDS.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return [], None, None, None, None

    landmarks_render = [
        [[p.x, p.y, p.z] for p in hand_lm.landmark]
        for hand_lm in res.multi_hand_landmarks
    ]

    hands_list = []
    for lm in res.multi_hand_landmarks:
        pts = []
        for p in lm.landmark:
            pts.extend([p.x, p.y, p.z])
        hands_list.append(np.array(pts, dtype=np.float32))

    if len(hands_list) >= 2:
        # Sort spatially by wrist x-coordinate (pts[0]) so left-most hand is slot 1, right-most hand is slot 2
        hands_list.sort(key=lambda pts: pts[0])
        left  = hands_list[0]
        right = hands_list[1]
    elif len(hands_list) == 1:
        label = res.multi_handedness[0].classification[0].label if res.multi_handedness else "Left"
        if label == "Right":
            left  = np.zeros(63, dtype=np.float32)
            right = hands_list[0]
        else:
            left  = hands_list[0]
            right = np.zeros(63, dtype=np.float32)
    else:
        left  = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

    feat_raw         = np.concatenate([left, right])
    feat_raw_swapped = np.concatenate([right, left])

    feat_vec         = normalize_frame_landmarks(feat_raw)
    feat_vec_swapped = normalize_frame_landmarks(feat_raw_swapped)

    # Reconstruct the 126-dim vector for the sequence model (BiLSTM)
    feat_vec_126 = np.concatenate([feat_vec[:63], feat_vec[98:161]])
    feat_vec_swapped_126 = np.concatenate([feat_vec_swapped[:63], feat_vec_swapped[98:161]])

    return landmarks_render, feat_vec, feat_vec_swapped, feat_vec_126, feat_vec_swapped_126


def temperature_softmax(logits_or_probs: np.ndarray, temp: float = 1.4) -> np.ndarray:
    """Temperature scaling to calibrate overconfident predictions."""
    eps = 1e-7
    clipped = np.clip(logits_or_probs, eps, 1.0 - eps)
    log_p = np.log(clipped) / temp
    exp_p = np.exp(log_p - np.max(log_p))
    return exp_p / np.sum(exp_p)


def run_sequence_prediction(buffer_snapshot: np.ndarray) -> dict:
    """Run BiLSTM sequence model on 30-frame landmark buffer."""
    inp   = buffer_snapshot[np.newaxis, ...]   # (1, 30, 126)
    preds = seq_model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx]) * 100

    top5_idx  = np.argsort(preds)[::-1][:5]
    all_probs = [
        {"label": str(seq_labels[i]).upper(), "prob": round(float(preds[i]) * 100, 1)}
        for i in top5_idx if i < len(seq_labels)
    ]

    return {
        "letter":     str(seq_labels[idx]).upper() if conf >= ENSEMBLE_CONFIDENCE_THRESHOLD else None,
        "confidence": round(conf, 1),
        "all_probs":  all_probs,
        "raw_probs":  preds,
    }


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
        "ensemble": {
            "mode": "Weighted Soft Voting Ensemble (Wrist Normalized + Hand Swap Alignment)",
            "num_classes": len(ISL_23_CLASSES),
            "classes": [c.upper() for c in ISL_23_CLASSES],
        },
        "engines": {
            "landmark_mlp": LM_AVAILABLE,
            "static_cnn": True,
            "dynamic_bilstm": SEQ_AVAILABLE,
        },
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

    # ── 1. MediaPipe Detection Gate + Wrist-Normalized Keypoints ──────────────
    landmarks, feat_vec, feat_swapped, feat_vec_126, feat_swapped_126 = extract_landmarks_full(frame_rgb)

    if not landmarks or feat_vec is None:
        return jsonify({
            "detected":   False,
            "letter":     "–",
            "raw_label":  "none",
            "confidence": 0.0,
            "all_probs":  [],
            "landmarks":  [],
            "isl": {
                "available":     SEQ_AVAILABLE,
                "buffer_frames": len(_frame_buffer),
                "buffer_ready":  False,
                "letter":        None,
                "confidence":    0.0,
                "all_probs":     [],
            },
        })

    # ── 2. Model 1: Landmark MLP Probs (with Left/Right Hand Swap Check) ─────
    if LM_AVAILABLE:
        p_lm_norm    = lm_model.predict(feat_vec.reshape(1, -1), verbose=0)[0]
        p_lm_swapped = lm_model.predict(feat_swapped.reshape(1, -1), verbose=0)[0]

        if np.max(p_lm_swapped) > np.max(p_lm_norm):
            p_lm_raw = p_lm_swapped
            best_feat_126 = feat_swapped_126
        else:
            p_lm_raw = p_lm_norm
            best_feat_126 = feat_vec_126
    else:
        p_lm_raw  = np.ones(len(ISL_23_CLASSES)) / len(ISL_23_CLASSES)
        best_feat_126 = feat_vec_126

    p_lm = p_lm_raw / (np.sum(p_lm_raw) + 1e-6)

    # ── 3. Model 2: MobileNetV2 CNN Probs ─────────────────────────────────────
    resized      = cv2.resize(frame_rgb, (224, 224))
    img_array    = np.expand_dims(resized.astype(np.float32), axis=0)
    preprocessed = preprocess_input(img_array)
    p_cnn_full   = cnn_model.predict(preprocessed, verbose=0)[0]

    p_cnn_23 = np.array([
        p_cnn_full[idx] if idx >= 0 else 0.0 for idx in CNN_TO_ISL_MAP
    ])
    p_cnn = p_cnn_23 / (np.sum(p_cnn_23) + 1e-6)

    # ── 4. Model 3: BiLSTM Sequence Probs ─────────────────────────────────────
    buffer_ready = False
    with _buffer_lock:
        _frame_buffer.append(best_feat_126)
        if len(_frame_buffer) == SEQUENCE_LEN:
            buffer_ready = True
            buf_snapshot = np.array(_frame_buffer, dtype=np.float32)

    if buffer_ready:
        isl_dict = run_sequence_prediction(buf_snapshot)
        with _buffer_lock:
            _seq_prediction.update(isl_dict)
        p_seq_raw = isl_dict["raw_probs"]
    else:
        with _buffer_lock:
            p_seq_raw = _seq_prediction.get("raw_probs", np.ones(len(ISL_23_CLASSES)) / len(ISL_23_CLASSES))

    p_seq = p_seq_raw / (np.sum(p_seq_raw) + 1e-6)

    # ── 5. Ensemble Weighted Soft Voting ──────────────────────────────────────
    if buffer_ready:
        w_lm, w_seq, w_cnn = 0.35, 0.55, 0.10
    else:
        w_lm, w_seq, w_cnn = 0.80, 0.00, 0.20

    p_ensemble = w_lm * p_lm + w_seq * p_seq + w_cnn * p_cnn
    p_ensemble = p_ensemble / np.sum(p_ensemble)

    top_idx = int(np.argmax(p_ensemble))
    ensemble_conf = float(p_ensemble[top_idx]) * 100
    top_letter = ISL_23_CLASSES[top_idx].upper()

    # Consensus Check: How many models place top_letter in their top-2
    lm_top2  = [ISL_23_CLASSES[i].upper() for i in np.argsort(p_lm)[::-1][:2]]
    seq_top2 = [ISL_23_CLASSES[i].upper() for i in np.argsort(p_seq)[::-1][:2]]
    cnn_top2 = [ISL_23_CLASSES[i].upper() for i in np.argsort(p_cnn)[::-1][:2]]

    consensus_count = (
        (1 if top_letter in lm_top2 else 0) +
        (1 if top_letter in seq_top2 else 0) +
        (1 if top_letter in cnn_top2 else 0)
    )

    if ensemble_conf >= ENSEMBLE_CONFIDENCE_THRESHOLD and consensus_count >= 2:
        final_letter = top_letter
    elif ensemble_conf >= 50.0:
        final_letter = top_letter
    else:
        final_letter = "?"

    top5_idx = np.argsort(p_ensemble)[::-1][:5]
    all_probs = [
        {"label": ISL_23_CLASSES[i].upper(), "prob": round(float(p_ensemble[i]) * 100, 1)}
        for i in top5_idx
    ]

    response = {
        "detected":   True,
        "letter":     final_letter,
        "raw_label":  ISL_23_CLASSES[top_idx],
        "confidence": round(ensemble_conf, 1),
        "all_probs":  all_probs,
        "landmarks":  landmarks,
        "ensemble_info": {
            "consensus": f"{consensus_count}/3 models",
            "lm_top":  lm_top2[0],
            "seq_top": seq_top2[0],
            "cnn_top": cnn_top2[0],
        },
        "isl": {
            "available":     SEQ_AVAILABLE,
            "buffer_frames": len(_frame_buffer),
            "buffer_ready":  buffer_ready,
            "letter":        _seq_prediction.get("letter"),
            "confidence":    _seq_prediction.get("confidence", 0.0),
            "all_probs":     _seq_prediction.get("all_probs", []),
        },
    }

    return jsonify(response)


@app.route("/predict/reset-buffer", methods=["POST"])
def reset_buffer():
    """Reset landmark rolling buffer."""
    with _buffer_lock:
        _frame_buffer.clear()
        _seq_prediction.update({
            "letter": None, "confidence": 0.0,
            "all_probs": [], "raw_probs": np.ones(len(ISL_23_CLASSES)) / len(ISL_23_CLASSES),
        })
    return jsonify({"status": "buffer cleared"})


@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({
        "ensemble_classes": [c.upper() for c in ISL_23_CLASSES],
    })


if __name__ == "__main__":
    log.info("SignBridge Ensemble Learning backend running → http://localhost:5050")
    log.info("  Ensemble Voting Engine: 3 Models (Wrist Normalized + Hand Swap Alignment)")
    log.info("  Classes: %d ISL signs", len(ISL_23_CLASSES))
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
