import os
import sys
import random
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = "/Users/pranav/Desktop/SignBridge/dataset - Gesture Speech"
MODEL_DIR = os.path.join(BASE_DIR, "backend", "model")

LANDMARK_MODEL_PATH = os.path.join(MODEL_DIR, "signbridge_landmark_model.keras")
LANDMARK_LABEL_PATH = os.path.join(MODEL_DIR, "label_classes.npy")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
SEQ_MODEL_PATH = os.path.join(MODEL_DIR, "isl_sequence_model.keras")
SEQ_LABEL_PATH = os.path.join(MODEL_DIR, "isl_label_classes.npy")

random.seed(42)
np.random.seed(42)

# Load Models
print("Loading models...")
lm_model = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
lm_labels = list(np.load(LANDMARK_LABEL_PATH, allow_pickle=True))

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
cnn_labels = list("abcdefghijklmnopqrstuvwxyz") + ["{"]

seq_model = tf.keras.models.load_model(SEQ_MODEL_PATH)
seq_labels = list(np.load(SEQ_LABEL_PATH, allow_pickle=True))

ISL_23_CLASSES = [c.lower() for c in seq_labels]
print(f"Aligning on {len(ISL_23_CLASSES)} ISL gesture classes: {ISL_23_CLASSES}")

CNN_TO_ISL_MAP = []
for c in ISL_23_CLASSES:
    if c in cnn_labels:
        CNN_TO_ISL_MAP.append(cnn_labels.index(c))
    else:
        CNN_TO_ISL_MAP.append(-1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)

def normalize_frame_landmarks(feat_126: np.ndarray) -> np.ndarray:
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

def extract_landmarks_raw_and_norm(img_rgb):
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None, None
    hands_dict = {}
    for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handedness.classification[0].label
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        wrist = coords[0]
        hand_centered = coords - wrist
        scale = np.linalg.norm(hand_centered[9]) + 1e-6
        hand_norm = hand_centered / scale
        hands_dict[label] = hand_norm.flatten()
    
    left = hands_dict.get("Left", np.zeros(63, dtype=np.float32))
    right = hands_dict.get("Right", np.zeros(63, dtype=np.float32))
    feat_126 = np.concatenate([left, right])
    feat_196 = normalize_frame_landmarks(feat_126)
    return feat_126, feat_196

# Collect samples
samples = []
SAMPLES_PER_CLASS = 40
for cls in ISL_23_CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(cls_path):
        continue
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    chosen = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
    for fname in chosen:
        samples.append((cls, os.path.join(cls_path, fname)))

print(f"Collected {len(samples)} validation samples.")

mlp_preds_list = []
seq_preds_list = []
cnn_preds_list = []
true_labels_list = []

print("Running inference across validation set...")
for idx, (cls, fpath) in enumerate(samples):
    img = cv2.imread(fpath)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract landmarks
    feat_126, feat_196 = extract_landmarks_raw_and_norm(img_rgb)
    if feat_126 is None:
        continue
        
    # 1. MLP Predict
    mlp_p = lm_model.predict(feat_196.reshape(1, -1), verbose=0)[0]
    
    # 2. BiLSTM Predict (repeat frame 30 times)
    seq_in = np.tile(feat_126, (30, 1)).reshape(1, 30, 126)
    seq_p = seq_model.predict(seq_in, verbose=0)[0]
    
    # 3. CNN Predict
    resized = cv2.resize(img_rgb, (224, 224))
    cnn_in = np.expand_dims(resized.astype(np.float32), axis=0)
    cnn_in = preprocess_input(cnn_in)
    cnn_raw = cnn_model.predict(cnn_in, verbose=0)[0]
    
    # Align CNN output to 23 classes
    cnn_p = np.zeros(len(ISL_23_CLASSES))
    for i, mapped_idx in enumerate(CNN_TO_ISL_MAP):
        if mapped_idx != -1:
            cnn_p[i] = cnn_raw[mapped_idx]
    # Re-normalize aligned CNN probabilities
    if cnn_p.sum() > 0:
        cnn_p /= cnn_p.sum()
        
    mlp_preds_list.append(mlp_p)
    seq_preds_list.append(seq_p)
    cnn_preds_list.append(cnn_p)
    true_labels_list.append(ISL_23_CLASSES.index(cls))

hands.close()

P_mlp = np.array(mlp_preds_list)
P_seq = np.array(seq_preds_list)
P_cnn = np.array(cnn_preds_list)
Y_true = np.array(true_labels_list)

print(f"\nSuccessfully evaluated {len(Y_true)} samples.")

# Grid search optimal weights
best_acc = 0.0
best_weights = (0.45, 0.45, 0.10)

step = 0.01
for w_mlp in np.arange(0.0, 1.0 + step, step):
    for w_seq in np.arange(0.0, 1.0 - w_mlp + step, step):
        w_cnn = 1.0 - w_mlp - w_seq
        if w_cnn < -1e-5:
            continue
        w_cnn = max(0.0, w_cnn)
        
        # Calculate ensemble predictions
        P_ens = w_mlp * P_mlp + w_seq * P_seq + w_cnn * P_cnn
        pred_labels = np.argmax(P_ens, axis=1)
        acc = np.mean(pred_labels == Y_true)
        
        if acc > best_acc:
            best_acc = acc
            best_weights = (w_mlp, w_seq, w_cnn)

print("\n" + "="*50)
print(f"Best Ensemble Accuracy: {best_acc*100:.2f}%")
print(f"Optimal Weights: MLP={best_weights[0]:.3f}, BiLSTM={best_weights[1]:.3f}, CNN={best_weights[2]:.3f}")
print("="*50)

# Write optimal weights back to backend/app.py
app_path = os.path.join(BASE_DIR, "backend", "app.py")
with open(app_path, "r") as f:
    app_content = f.read()

# Replace weight definition line
# Example line in app.py: w_lm, w_seq, w_cnn = 0.45, 0.45, 0.10
import re
new_weights_str = f"w_lm, w_seq, w_cnn = {best_weights[0]:.3f}, {best_weights[1]:.3f}, {best_weights[2]:.3f}"
app_content_updated, count = re.subn(
    r"w_lm,\s*w_seq,\s*w_cnn\s*=\s*[0-9.]+,\s*[0-9.]+,\s*[0-9.]+",
    new_weights_str,
    app_content
)

if count > 0:
    with open(app_path, "w") as f:
        f.write(app_content_updated)
    print(f"Updated {app_path} with new weights: {new_weights_str}")
else:
    print("WARNING: Could not find weight declaration line in backend/app.py to auto-update.")
