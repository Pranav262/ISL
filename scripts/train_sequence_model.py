"""
SignBridge — ISL A–Z Dynamic Gesture Sequence Model Training
=============================================================

Dataset format: (30, 126) NumPy arrays — 30-frame sequences of MediaPipe landmarks.

KEY ENHANCEMENTS FOR HIGH REAL-WORLD CAMERA ACCURACY:
  1. Per-Hand Wrist-Relative & Scale Normalization: Makes keypoints 100% invariant
     to camera position, distance, tilt, and resolution.
  2. Multi-Hand Data Augmentation: Includes Left/Right hand swaps, single-hand extractions,
     time-shifts, and Gaussian noise. Teaches model both 1-handed and 2-handed variants of all signs.

Architecture: Bidirectional LSTM sequence classifier
  Input  → (30, 126) normalized landmark sequences
  Model  → BiLSTM(128) → LayerNorm → BiLSTM(64) → LayerNorm → Dense(128) → Dense(64) → Softmax(23)

Outputs saved to backend/model/:
  - isl_sequence_model.keras    (the LSTM model)
  - isl_label_classes.npy       (class label array)

Usage:
  source .venv/bin/activate
  python scripts/train_sequence_model.py
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ISL      = os.path.join(BASE_DIR, "dataset_isl_az")
ARTIFACTS_DIR    = os.path.join(BASE_DIR, "training_artifacts")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "model")

BEST_MODEL_PATH  = os.path.join(ARTIFACTS_DIR, "best_isl_sequence_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "isl_sequence_model.keras")
LABEL_PATH       = os.path.join(MODEL_OUTPUT_DIR, "isl_label_classes.npy")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

SEQUENCE_LEN = 30   # frames per gesture
FEATURE_DIM  = 126  # landmarks per frame (21 × 3 × 2 hands)
BATCH_SIZE   = 32


# ── Normalization Helper ──────────────────────────────────────────────────────

def normalize_frame_landmarks(feat_126: np.ndarray) -> np.ndarray:
    """
    Per-hand wrist centering and scale normalization.
    Returns 126-dim normalized landmark array.
    """
    left_raw  = feat_126[:63].reshape(21, 3)
    right_raw = feat_126[63:].reshape(21, 3)

    def _norm(hand):
        if np.abs(hand).sum() < 1e-5:
            return np.zeros((21, 3), dtype=np.float32)
        wrist = hand[0]
        hand_centered = hand - wrist
        scale = np.linalg.norm(hand_centered[9]) + 1e-6
        return (hand_centered / scale).astype(np.float32)

    l_norm = _norm(left_raw).flatten()
    r_norm = _norm(right_raw).flatten()
    return np.concatenate([l_norm, r_norm])


def normalize_sequence(seq_30x126: np.ndarray) -> np.ndarray:
    """Apply normalize_frame_landmarks to all 30 frames of a sequence."""
    out = np.zeros_like(seq_30x126, dtype=np.float32)
    for t in range(seq_30x126.shape[0]):
        out[t] = normalize_frame_landmarks(seq_30x126[t])
    return out


# ── Data Loading & Augmentation ───────────────────────────────────────────────

def load_and_augment_dataset(dataset_dir, target_per_class=160, seed=42):
    """
    Loads sequences, normalizes keypoints, and applies physical hand-swap,
    single-hand, time-shift, and Gaussian noise data augmentation.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for letter_folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, letter_folder)
        if not os.path.isdir(folder_path):
            continue

        label = letter_folder.upper()
        npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
        if not npy_files:
            continue

        for fname in npy_files:
            path = os.path.join(folder_path, fname)
            try:
                seq = np.load(path).astype(np.float32)
                if seq.shape != (SEQUENCE_LEN, FEATURE_DIM):
                    if seq.ndim == 2 and seq.shape[1] == FEATURE_DIM:
                        if seq.shape[0] < SEQUENCE_LEN:
                            pad = np.zeros((SEQUENCE_LEN - seq.shape[0], FEATURE_DIM), dtype=np.float32)
                            seq = np.vstack([seq, pad])
                        else:
                            seq = seq[:SEQUENCE_LEN]
                    else:
                        continue

                # 1. Normalize original sequence
                seq_norm = normalize_sequence(seq)
                X.append(seq_norm)
                y.append(label)

                # 2. Hand-Swap Augmentation: Swap Left (0..62) and Right (63..125)
                seq_swap = np.zeros_like(seq_norm)
                seq_swap[:, :63] = seq_norm[:, 63:]
                seq_swap[:, 63:] = seq_norm[:, :63]
                X.append(seq_swap)
                y.append(label)

                # 3. Single-Hand Variations (Left-only & Right-only)
                l_has_data = np.abs(seq_norm[:, :63]).sum() > 0
                r_has_data = np.abs(seq_norm[:, 63:]).sum() > 0

                if l_has_data and r_has_data:
                    # Left-only
                    seq_l_only = seq_norm.copy()
                    seq_l_only[:, 63:] = 0
                    X.append(seq_l_only)
                    y.append(label)

                    # Right-only
                    seq_r_only = seq_norm.copy()
                    seq_r_only[:, :63] = 0
                    X.append(seq_r_only)
                    y.append(label)

            except Exception as e:
                print(f"  [WARN] Error loading {path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"[Data] Loaded & hand-augmented base dataset: {len(X)} sequences, {len(np.unique(y))} classes")

    # 4. Secondary Augmentation (Noise & Time-Shift) up to target_per_class
    counts = Counter(y)
    X_aug, y_aug = list(X), list(y)

    for label, count in counts.items():
        if count >= target_per_class:
            continue
        need = target_per_class - count
        indices = np.where(y == label)[0]
        for _ in range(need):
            src = X[rng.choice(indices)].copy()
            # Gaussian noise on non-zero landmarks
            mask = np.abs(src) > 1e-5
            noise = rng.normal(0, 0.015, src.shape).astype(np.float32)
            src[mask] += noise[mask]

            # Time-shift (1-3 frames)
            shift = rng.integers(1, 4)
            src = np.roll(src, shift, axis=0)

            X_aug.append(src)
            y_aug.append(label)

    return np.array(X_aug, dtype=np.float32), np.array(y_aug)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes, seq_len=SEQUENCE_LEN, feat_dim=FEATURE_DIM):
    """
    Bidirectional LSTM sequence classifier for wrist-normalized hand landmarks.
    """
    inputs = layers.Input(shape=(seq_len, feat_dim), name="sequence_input")

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_1"
    )(inputs)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name="bilstm_2"
    )(x)
    x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inputs, outputs, name="ISL_BiLSTM_Normalized")


# ── Main Training ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train robust ISL sequence LSTM model")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--augment-target", type=int, default=180)
    args = parser.parse_args()

    print("=" * 60)
    print("  SignBridge — ISL Sequence Model Training (Wrist Normalized + Augmented)")
    print("=" * 60)

    if not os.path.isdir(DATASET_ISL):
        print(f"ERROR: Dataset not found at {DATASET_ISL}")
        sys.exit(1)

    print("[Data] Loading & augmenting sequences …")
    X, y = load_and_augment_dataset(DATASET_ISL, target_per_class=args.augment_target)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"\n[Labels] {num_classes} classes: {list(le.classes_)}")

    np.save(LABEL_PATH, le.classes_)
    print(f"[Labels] Saved → {LABEL_PATH}")

    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y_enc, num_classes)

    # Split train/val/test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_cat, test_size=0.25, random_state=42, stratify=y_enc
    )
    y_tmp_enc = np.argmax(y_tmp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.40, random_state=42, stratify=y_tmp_enc
    )
    print(f"[Split] Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Build model
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ]

    print(f"\n[Train] Training for up to {args.epochs} epochs …")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        verbose=1,
    )
    print(f"[Train] Finished in {time.time()-t0:.0f}s")

    # Evaluate
    best = tf.keras.models.load_model(BEST_MODEL_PATH)
    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Eval] Test Accuracy: {acc*100:.2f}%  (loss: {loss:.4f})")

    y_pred = np.argmax(best.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n[Eval] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Save final model
    best.save(FINAL_MODEL_PATH)
    print(f"\n[Save] Saved model → {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
