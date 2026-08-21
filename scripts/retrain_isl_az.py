"""
SignBridge — ISL A–Z Full Retrain Script
=========================================
Retrains BOTH models using the combined dataset:
  1. Existing "dataset - Gesture Speech" (synthetic a-z + '{' for SPACE)
  2. New ISL A-Z dataset  (real photos, folders A-Z)

Outputs (saved to backend/model/):
  - signbridge_landmark_model.keras   (Landmark MLP, 126-dim MediaPipe input)
  - best_model.keras                  (MobileNetV2 CNN, 224x224 image input)
  - label_classes.npy                 (class label array, shape [27])

Usage:
  # From the SignBridge project root:
  source .venv/bin/activate
  python scripts/retrain_isl_az.py

  # To skip MobileNetV2 training (faster, landmark model only):
  python scripts/retrain_isl_az.py --landmark-only

  # To skip landmark training (CNN only):
  python scripts/retrain_isl_az.py --cnn-only
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SYNTH    = os.path.join(BASE_DIR, "dataset - Gesture Speech")
DATASET_ISL      = os.path.join(BASE_DIR, "dataset_isl_az")
ARTIFACTS_DIR    = os.path.join(BASE_DIR, "training_artifacts")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "model")

# Artifact paths
LANDMARK_NPZ_PATH   = os.path.join(ARTIFACTS_DIR, "landmark_combined.npz")
BEST_LANDMARK_PATH  = os.path.join(ARTIFACTS_DIR, "best_landmark_model.keras")
FINAL_LANDMARK_PATH = os.path.join(MODEL_OUTPUT_DIR, "signbridge_landmark_model.keras")
BEST_CNN_PATH       = os.path.join(ARTIFACTS_DIR, "best_mobilenetv2.keras")
FINAL_CNN_PATH      = os.path.join(MODEL_OUTPUT_DIR, "best_model.keras")
LABEL_CLASSES_PATH  = os.path.join(MODEL_OUTPUT_DIR, "label_classes.npy")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands

# ── Class config ─────────────────────────────────────────────────────────────
# The combined class set is a-z (26 letters) + '{' (SPACE)
# We normalise folder names to lowercase for consistent merging.
ALL_CLASSES = list("abcdefghijklmnopqrstuvwxyz") + ["{"]   # 27 classes


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_two_hand_landmarks(img_rgb, hands_detector):
    """Returns a 126-dim feature vector or None if no hand detected."""
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None

    hands_dict = {}
    for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handedness.classification[0].label   # "Left" / "Right"
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        # Normalize: centre on wrist, scale by wrist-to-middle-MCP distance
        wrist = coords[0]
        coords = coords - wrist
        scale = np.linalg.norm(coords[9]) + 1e-6
        coords = coords / scale
        hands_dict[label] = coords.flatten()   # 21 * 3 = 63 values

    left  = hands_dict.get("Left",  np.zeros(63, dtype=np.float32))
    right = hands_dict.get("Right", np.zeros(63, dtype=np.float32))
    return np.concatenate([left, right])   # 126-dim


def collect_image_paths(dataset_dirs, valid_classes):
    """
    Walk one or more dataset root directories and collect (label, path) pairs.
    `valid_classes` is a set of lowercase class names to include.
    """
    pairs = []
    for root in dataset_dirs:
        if not os.path.isdir(root):
            print(f"  [WARN] Dataset dir not found, skipping: {root}")
            continue
        for folder in sorted(os.listdir(root)):
            folder_lower = folder.lower()
            if folder_lower not in valid_classes:
                continue
            class_dir = os.path.join(root, folder)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                pairs.append((folder_lower, os.path.join(class_dir, fname)))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — LANDMARK MLP
# ══════════════════════════════════════════════════════════════════════════════

def build_landmark_dataset(image_pairs, out_npz_path, save_every=3000):
    """Extract MediaPipe landmarks from images; cache result to .npz."""
    print("\n[Landmark] Extracting hand landmarks …")
    X, y, skipped = [], [], 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    ) as hands:
        for i, (label, fpath) in enumerate(tqdm(image_pairs, desc="Extracting")):
            img_rgb = load_image_rgb(fpath)
            if img_rgb is None:
                skipped += 1
                continue
            feat = extract_two_hand_landmarks(img_rgb, hands)
            if feat is None:
                skipped += 1
                continue
            X.append(feat)
            y.append(label)

            if (i + 1) % save_every == 0:
                np.savez(out_npz_path, X=np.array(X), y=np.array(y))

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    np.savez(out_npz_path, X=X, y=y)
    total = len(image_pairs)
    pct   = len(X) / total * 100 if total else 0
    print(f"[Landmark] Done. Extracted {len(X)}/{total} samples ({pct:.1f}%). Skipped {skipped}.")
    return X, y


def train_landmark_model(image_pairs, force_reextract=False):
    print("\n" + "="*60)
    print("  LANDMARK MLP — Training")
    print("="*60)

    # ── Feature extraction (or load cached) ──────────────────────────
    if os.path.exists(LANDMARK_NPZ_PATH) and not force_reextract:
        print(f"[Landmark] Loading cached features from {LANDMARK_NPZ_PATH}")
        data = np.load(LANDMARK_NPZ_PATH, allow_pickle=True)
        X, y_labels = data["X"], data["y"]
    else:
        X, y_labels = build_landmark_dataset(image_pairs, LANDMARK_NPZ_PATH)

    # ── Encode labels ─────────────────────────────────────────────────
    le          = LabelEncoder()
    y_encoded   = le.fit_transform(y_labels)
    num_classes = len(le.classes_)
    print(f"[Landmark] Classes ({num_classes}): {list(le.classes_)}")

    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y_encoded)

    # ── Train/val/test split ──────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_cat, test_size=0.30, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42,
        stratify=np.argmax(y_tmp, axis=1)
    )
    print(f"[Landmark] Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

    # ── Build model ───────────────────────────────────────────────────
    model = models.Sequential([
        layers.Input(shape=(126,)),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_LANDMARK_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=64,
        callbacks=cbs,
    )
    print(f"[Landmark] Training finished in {time.time()-t0:.0f}s")

    # ── Evaluate on test set ──────────────────────────────────────────
    best = tf.keras.models.load_model(BEST_LANDMARK_PATH)
    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Landmark] Test accuracy: {acc*100:.2f}%  (loss {loss:.4f})")

    y_pred   = np.argmax(best.predict(X_test, verbose=0), axis=1)
    y_true   = np.argmax(y_test, axis=1)
    print("\n[Landmark] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # ── Save to model output dir ──────────────────────────────────────
    best.save(FINAL_LANDMARK_PATH)
    print(f"[Landmark] Saved → {FINAL_LANDMARK_PATH}")

    return best, le


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — MOBILENETV2 CNN
# ══════════════════════════════════════════════════════════════════════════════

IMG_SIZE   = 224
BATCH_SIZE = 32


def build_image_dataset(image_pairs, label_encoder, img_size=IMG_SIZE):
    """Build TF dataset from (label, path) pairs."""

    labels_encoded = label_encoder.transform([lbl for lbl, _ in image_pairs])
    paths          = [p for _, p in image_pairs]

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_encoded))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def build_augmentation_layer():
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def build_mobilenetv2_model(num_classes):
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze during Phase 1

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = build_augmentation_layer()(inputs, training=True)
    x       = preprocess_input(x)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base


def train_cnn_model(image_pairs, label_encoder):
    print("\n" + "="*60)
    print("  MOBILENETV2 CNN — Training")
    print("="*60)

    num_classes = len(label_encoder.classes_)
    print(f"[CNN] Classes ({num_classes}): {list(label_encoder.classes_)}")

    # ── Split indices ──────────────────────────────────────────────────
    labels_all = [lbl for lbl, _ in image_pairs]
    labels_enc = label_encoder.transform(labels_all)

    idx_all = np.arange(len(image_pairs))
    idx_train, idx_tmp = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=labels_enc)
    labels_tmp = labels_enc[idx_tmp]
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=labels_tmp)

    pairs_train = [image_pairs[i] for i in idx_train]
    pairs_val   = [image_pairs[i] for i in idx_val]
    pairs_test  = [image_pairs[i] for i in idx_test]

    print(f"[CNN] Train {len(pairs_train)}  Val {len(pairs_val)}  Test {len(pairs_test)}")

    # ── Build TF datasets ──────────────────────────────────────────────
    ds_train = (build_image_dataset(pairs_train, label_encoder)
                .shuffle(2000)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    ds_val   = (build_image_dataset(pairs_val, label_encoder)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    ds_test  = (build_image_dataset(pairs_test, label_encoder)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    # ── Build model ────────────────────────────────────────────────────
    model, base_model = build_mobilenetv2_model(num_classes)

    # ── PHASE 1: Train head only ───────────────────────────────────────
    print("\n[CNN] Phase 1 — training classification head (base frozen)…")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cbs_p1 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_CNN_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]
    t0 = time.time()
    model.fit(ds_train, validation_data=ds_val, epochs=30, callbacks=cbs_p1)
    print(f"[CNN] Phase 1 done in {time.time()-t0:.0f}s")

    # ── PHASE 2: Fine-tune last 30 layers of base ─────────────────────
    print("\n[CNN] Phase 2 — fine-tuning last 30 layers of MobileNetV2…")
    best_p1 = tf.keras.models.load_model(BEST_CNN_PATH)
    # Re-attach base; unfreeze last 30 layers
    for layer in best_p1.layers:
        if hasattr(layer, "layers"):  # it's the MobileNetV2 base
            for bl in layer.layers[:-30]:
                bl.trainable = False
            for bl in layer.layers[-30:]:
                bl.trainable = True

    best_p1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs_p2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_CNN_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
    ]
    t0 = time.time()
    best_p1.fit(ds_train, validation_data=ds_val, epochs=50, callbacks=cbs_p2)
    print(f"[CNN] Phase 2 done in {time.time()-t0:.0f}s")

    # ── Evaluate on test set ───────────────────────────────────────────
    final = tf.keras.models.load_model(BEST_CNN_PATH)
    loss, acc = final.evaluate(ds_test, verbose=0)
    print(f"\n[CNN] Test accuracy: {acc*100:.2f}%  (loss {loss:.4f})")

    # Generate Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    y_preds, y_trues = [], []
    for x_b, y_b in ds_test:
        p_b = final.predict(x_b, verbose=0)
        y_preds.extend(np.argmax(p_b, axis=1))
        y_trues.extend(y_b.numpy())
    
    cm = confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('MobileNetV2 CNN Confusion Matrix (Test Set)')
    plt.tight_layout()
    cnn_cm_path = os.path.join(ARTIFACTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cnn_cm_path, dpi=300)
    print(f"[CNN] Confusion matrix saved → {cnn_cm_path}")

    # ── Save to model output dir ───────────────────────────────────────
    final.save(FINAL_CNN_PATH)
    print(f"[CNN] Saved → {FINAL_CNN_PATH}")

    return final


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SignBridge ISL A-Z retraining script")
    parser.add_argument("--landmark-only", action="store_true", help="Train only the Landmark MLP model")
    parser.add_argument("--cnn-only",      action="store_true", help="Train only the MobileNetV2 CNN model")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Force re-extraction of landmarks (ignore cache)")
    args = parser.parse_args()

    do_landmark = not args.cnn_only
    do_cnn      = not args.landmark_only

    print("="*60)
    print("  SignBridge — ISL A-Z Combined Retraining")
    print("="*60)
    print(f"  Existing dataset  : {DATASET_SYNTH}")
    print(f"  New ISL dataset   : {DATASET_ISL}")
    print(f"  Landmark MLP      : {'YES' if do_landmark else 'SKIP'}")
    print(f"  MobileNetV2 CNN   : {'YES' if do_cnn else 'SKIP'}")
    print(f"  Artifacts dir     : {ARTIFACTS_DIR}")
    print(f"  Model output dir  : {MODEL_OUTPUT_DIR}")
    print()

    # ── Check datasets exist ──────────────────────────────────────────
    found_any = False
    for d in [DATASET_SYNTH, DATASET_ISL]:
        if os.path.isdir(d):
            print(f"  [OK] {d}")
            found_any = True
        else:
            print(f"  [WARN] Not found: {d}")
    if not found_any:
        print("\nERROR: No dataset directories found. Aborting.")
        sys.exit(1)

    # ── Collect image paths ───────────────────────────────────────────
    valid_classes = set(ALL_CLASSES)
    valid_classes.add("{")  # include SPACE class

    dataset_dirs = [d for d in [DATASET_SYNTH, DATASET_ISL] if os.path.isdir(d)]
    image_pairs  = collect_image_paths(dataset_dirs, valid_classes)
    print(f"\n[Data] Total images collected: {len(image_pairs)}")

    # Print per-class counts
    from collections import Counter
    counts = Counter(lbl for lbl, _ in image_pairs)
    for cls in sorted(counts):
        print(f"  {cls:>3s}: {counts[cls]:>5d} images")

    if len(image_pairs) == 0:
        print("\nERROR: No images found. Check dataset directories.")
        sys.exit(1)

    # ── Build shared label encoder ────────────────────────────────────
    # Use only classes that actually have images
    actual_classes = sorted(counts.keys())
    le = LabelEncoder()
    le.fit(actual_classes)
    print(f"\n[Labels] Encoding {len(le.classes_)} classes: {list(le.classes_)}")

    # Save label classes immediately (so backend can be updated)
    np.save(LABEL_CLASSES_PATH, le.classes_)
    print(f"[Labels] Saved → {LABEL_CLASSES_PATH}")

    # ── Train models ──────────────────────────────────────────────────
    if do_landmark:
        train_landmark_model(image_pairs, force_reextract=args.force_reextract)

    if do_cnn:
        train_cnn_model(image_pairs, le)

    print("\n" + "="*60)
    print("  ALL DONE — Models saved to backend/model/")
    print("="*60)
    print("  Restart the backend server to use new models:")
    print("    cd backend && python app.py")


if __name__ == "__main__":
    main()
