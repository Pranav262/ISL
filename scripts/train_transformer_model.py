"""
SignBridge — ISL A–Z Transformer Encoder Sequence Model Experiment
===================================================================

Replaces BiLSTM with a Transformer Encoder architecture (Multi-Head Self-Attention)
for 30-frame MediaPipe landmark sequences.
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
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ISL = os.path.join(BASE_DIR, "dataset_isl_az")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "training_artifacts")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "model")

BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_transformer_model.keras")
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "isl_transformer_model.keras")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "transformer_confusion_matrix.png")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

SEQUENCE_LEN = 30
FEATURE_DIM = 126
BATCH_SIZE = 32

def normalize_frame_landmarks(feat_126: np.ndarray) -> np.ndarray:
    left_raw = feat_126[:63].reshape(21, 3)
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
    out = np.zeros_like(seq_30x126, dtype=np.float32)
    for t in range(seq_30x126.shape[0]):
        out[t] = normalize_frame_landmarks(seq_30x126[t])
    return out

def load_and_augment_dataset(dataset_dir, target_per_class=180, seed=42):
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

                seq_norm = normalize_sequence(seq)
                X.append(seq_norm)
                y.append(label)

                # Hand Swap
                seq_swap = np.zeros_like(seq_norm)
                seq_swap[:, :63] = seq_norm[:, 63:]
                seq_swap[:, 63:] = seq_norm[:, :63]
                X.append(seq_swap)
                y.append(label)

            except Exception as e:
                pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    counts = Counter(y)
    X_aug, y_aug = list(X), list(y)
    for label, count in counts.items():
        if count >= target_per_class: continue
        need = target_per_class - count
        indices = np.where(y == label)[0]
        for _ in range(need):
            src = X[rng.choice(indices)].copy()
            mask = np.abs(src) > 1e-5
            noise = rng.normal(0, 0.015, src.shape).astype(np.float32)
            src[mask] += noise[mask]
            shift = rng.integers(1, 4)
            src = np.roll(src, shift, axis=0)
            X_aug.append(src)
            y_aug.append(label)

    return np.array(X_aug, dtype=np.float32), np.array(y_aug)

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, seq_len=SEQUENCE_LEN, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model,
        })
        return config

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Self Attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)

    # Feed Forward
    y = layers.Dense(ff_dim, activation="relu")(x)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    res2 = y + x
    return layers.LayerNormalization(epsilon=1e-6)(res2)

def build_transformer_model(num_classes, seq_len=SEQUENCE_LEN, feat_dim=FEATURE_DIM):
    inputs = layers.Input(shape=(seq_len, feat_dim))
    
    # Project features to d_model
    d_model = 128
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)

    # 2 Transformer Encoder Blocks
    x = transformer_encoder_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder_block(x, head_size=32, num_heads=4, ff_dim=128, dropout=0.2)

    # Pooling & Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="ISL_Transformer")

def main():
    print("=" * 60)
    print("  SignBridge — Training Transformer Sequence Model")
    print("=" * 60)

    X, y = load_and_augment_dataset(DATASET_ISL, target_per_class=180)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_enc, num_classes)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_cat, test_size=0.25, random_state=42, stratify=y_enc)
    y_tmp_enc = np.argmax(y_tmp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.40, random_state=42, stratify=y_tmp_enc)

    model = build_transformer_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=cbs)
    print(f"\n[Train] Finished in {time.time()-t0:.0f}s")

    best = tf.keras.models.load_model(BEST_MODEL_PATH, custom_objects={"PositionalEncoding": PositionalEncoding})
    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Eval] Test Accuracy: {acc*100:.2f}%  (loss: {loss:.4f})")

    y_pred = np.argmax(best.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n[Eval] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Transformer Confusion Matrix (Validation)')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    print(f"[Save] Confusion matrix saved → {CONFUSION_MATRIX_PATH}")

    best.save(FINAL_MODEL_PATH)

if __name__ == "__main__":
    main()
