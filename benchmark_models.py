"""
SignBridge — Model Benchmark
Tests both models on a random sample from the dataset and reports:
  - Per-class accuracy
  - Overall accuracy
  - Average inference latency
"""

import os
import sys
import time
import random
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = "/Users/pranav/Desktop/SignBridge/dataset - Gesture Speech"
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "backend", "model")
LANDMARK_PATH = os.path.join(MODEL_DIR, "signbridge_landmark_model.keras")
MOBILENET_PATH= os.path.join(MODEL_DIR, "best_model.keras")
LABEL_PATH    = os.path.join(MODEL_DIR, "label_classes.npy")

SAMPLES_PER_CLASS = 30   # How many images to test per class
IMG_SIZE          = (224, 224)
CONFIDENCE_THRESHOLD = 40.0

random.seed(42)

# ── Load dataset class dirs ───────────────────────────────────────────────────
class_dirs = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
])
print(f"Found {len(class_dirs)} classes: {class_dirs}\n")

# Sample files
test_files = []   # list of (true_label, filepath)
for cls in class_dirs:
    cls_path = os.path.join(DATASET_DIR, cls)
    images   = [
        f for f in os.listdir(cls_path)
        if not f.startswith(".") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    chosen = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
    for fname in chosen:
        test_files.append((cls, os.path.join(cls_path, fname)))

print(f"Total test samples: {len(test_files)}\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ── Benchmark 1: Landmark DNN ─────────────────────────────────────────────────
print("=" * 60)
print("🔬  Model 1: MediaPipe Landmark DNN")
print("=" * 60)

label_classes_lm = np.load(LABEL_PATH, allow_pickle=True)
landmark_model   = tf.keras.models.load_model(LANDMARK_PATH)
mp_hands         = mp.solutions.hands
hands            = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
)

def extract_landmarks(img_rgb):
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None
    hands_dict = {}
    for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        label  = handedness.classification[0].label
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        wrist  = coords[0]
        coords = (coords - wrist) / (np.linalg.norm(coords[9]) + 1e-6)
        hands_dict[label] = coords.flatten()
    left  = hands_dict.get("Left",  np.zeros(63, dtype=np.float32))
    right = hands_dict.get("Right", np.zeros(63, dtype=np.float32))
    return np.concatenate([left, right])

lm_correct = lm_skipped = 0
lm_latencies = []
lm_per_class = {c: {"correct": 0, "total": 0} for c in class_dirs}

print("Running... (this may take a minute)")
for true_label, fpath in test_files:
    img_rgb = load_rgb(fpath)
    if img_rgb is None:
        lm_skipped += 1
        continue

    t0   = time.perf_counter()
    feat = extract_landmarks(img_rgb)
    if feat is None:
        lm_skipped += 1
        continue
    preds = landmark_model.predict(feat.reshape(1, -1), verbose=0)[0]
    t1    = time.perf_counter()

    lm_latencies.append((t1 - t0) * 1000)
    idx        = int(np.argmax(preds))
    pred_label = str(label_classes_lm[idx])
    lm_per_class[true_label]["total"] += 1
    if pred_label == true_label:
        lm_correct += 1
        lm_per_class[true_label]["correct"] += 1

hands.close()
lm_total = len(test_files) - lm_skipped
lm_acc   = lm_correct / max(lm_total, 1) * 100

print(f"\n  ✅ Overall Accuracy : {lm_acc:.1f}%")
print(f"  ⏱  Avg Latency     : {np.mean(lm_latencies):.1f} ms/frame")
print(f"  🚫 Skipped (no hand): {lm_skipped} / {len(test_files)}")

print("\n  Per-class accuracy:")
for cls in class_dirs:
    d = lm_per_class[cls]
    if d["total"] > 0:
        acc = d["correct"] / d["total"] * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {cls:3s}  [{bar}] {acc:5.1f}%")

# ── Benchmark 2: MobileNetV2 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🔬  Model 2: MobileNetV2 (Image CNN)")
print("=" * 60)

mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
# Class names come from dataset folder order (same as training)
mv2_classes = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
])
print(f"  MobileNetV2 classes: {mv2_classes}")

mv2_correct = mv2_skipped = 0
mv2_latencies = []
mv2_per_class = {c: {"correct": 0, "total": 0} for c in class_dirs}

print("Running...")
for true_label, fpath in test_files:
    img_rgb = load_rgb(fpath)
    if img_rgb is None:
        mv2_skipped += 1
        continue

    t0      = time.perf_counter()
    resized = cv2.resize(img_rgb, IMG_SIZE)
    arr     = np.expand_dims(resized.astype(np.float32), axis=0)
    arr     = preprocess_input(arr)
    preds   = mobilenet_model.predict(arr, verbose=0)[0]
    t1      = time.perf_counter()

    mv2_latencies.append((t1 - t0) * 1000)
    idx        = int(np.argmax(preds))
    pred_label = mv2_classes[idx] if idx < len(mv2_classes) else "?"
    mv2_per_class[true_label]["total"] += 1
    if pred_label == true_label:
        mv2_correct += 1
        mv2_per_class[true_label]["correct"] += 1

mv2_total = len(test_files) - mv2_skipped
mv2_acc   = mv2_correct / max(mv2_total, 1) * 100

print(f"\n  ✅ Overall Accuracy : {mv2_acc:.1f}%")
print(f"  ⏱  Avg Latency     : {np.mean(mv2_latencies):.1f} ms/frame")
print(f"  🚫 Skipped         : {mv2_skipped}")

print("\n  Per-class accuracy:")
for cls in class_dirs:
    d = mv2_per_class[cls]
    if d["total"] > 0:
        acc = d["correct"] / d["total"] * 100
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {cls:3s}  [{bar}] {acc:5.1f}%")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("🏆  BENCHMARK SUMMARY")
print("=" * 60)
print(f"  MediaPipe Landmark DNN : {lm_acc:.1f}% accuracy  |  {np.mean(lm_latencies):.1f} ms avg")
print(f"  MobileNetV2 CNN        : {mv2_acc:.1f}% accuracy  |  {np.mean(mv2_latencies):.1f} ms avg")

best = "landmark" if lm_acc >= mv2_acc else "mobilenet"
print(f"\n  → Best model for accuracy: {'MediaPipe Landmark DNN' if best == 'landmark' else 'MobileNetV2'}")
print(f"  → Best model for speed:    {'MediaPipe Landmark DNN' if np.mean(lm_latencies) <= np.mean(mv2_latencies) else 'MobileNetV2'}")

# Write result to file for the backend to read
result = {
    "landmark_accuracy": round(lm_acc, 2),
    "mobilenet_accuracy": round(mv2_acc, 2),
    "landmark_latency_ms": round(float(np.mean(lm_latencies)), 2),
    "mobilenet_latency_ms": round(float(np.mean(mv2_latencies)), 2),
    "recommended": best,
}
import json
out = os.path.join(os.path.dirname(__file__), "benchmark_result.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved to benchmark_result.json")
print("=" * 60)
