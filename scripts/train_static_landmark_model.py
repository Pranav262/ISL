import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ISL = os.path.join(BASE_DIR, "dataset_isl_az")
LANDMARK_NPZ = os.path.join(BASE_DIR, "artifacts", "landmark_dataset.npz")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "model")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "training_artifacts")

BEST_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_landmark_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "signbridge_landmark_model.keras")
LABEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "label_classes.npy")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "landmark_confusion_matrix.png")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

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

def load_all_static_landmarks():
    X_list, y_list = [], []
    if os.path.exists(LANDMARK_NPZ):
        print(f"[Data] Loading synthetic dataset from {LANDMARK_NPZ} …")
        d = np.load(LANDMARK_NPZ, allow_pickle=True)
        X_synth, y_synth = d["X"], d["y"]
        X_synth_norm = np.array([normalize_frame_landmarks(f) for f in X_synth], dtype=np.float32)
        X_list.append(X_synth_norm)
        y_list.append(np.array([s.lower() for s in y_synth]))
        print(f"  Loaded & engineered {len(X_synth)} synthetic samples.")

    if os.path.isdir(DATASET_ISL):
        print(f"[Data] Extracting & augmenting static frames from {DATASET_ISL} …")
        X_isl, y_isl = [], []
        for letter in sorted(os.listdir(DATASET_ISL)):
            ld = os.path.join(DATASET_ISL, letter)
            if not os.path.isdir(ld): continue
            lbl = letter.lower()
            for f in os.listdir(ld):
                if f.endswith(".npy"):
                    seq = np.load(os.path.join(ld, f))
                    if seq.ndim == 2 and seq.shape[1] == 126:
                        for frame in seq:
                            if np.abs(frame).sum() > 0:
                                norm_f = normalize_frame_landmarks(frame)
                                X_isl.append(norm_f)
                                y_isl.append(lbl)
                                
                                # Hand swap logic: feat is now 196 (98 + 98)
                                swap_f = np.zeros_like(norm_f)
                                swap_f[:98] = norm_f[98:]
                                swap_f[98:] = norm_f[:98]
                                X_isl.append(swap_f)
                                y_isl.append(lbl)
        X_isl = np.array(X_isl, dtype=np.float32)
        y_isl = np.array(y_isl)
        print(f"  Extracted & augmented {len(X_isl)} static frames from ISL dataset.")
        X_list.append(X_isl)
        y_list.append(y_isl)

    if not X_list: raise FileNotFoundError("No datasets found!")
    return np.vstack(X_list), np.concatenate(y_list)

def build_landmark_mlp(num_classes):
    model = models.Sequential([
        layers.Input(shape=(196,)),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def main():
    print("=" * 60)
    print("  SignBridge — Retraining Static Landmark MLP Model (Engineered Features + Reg)")
    print("=" * 60)

    X, y = load_all_static_landmarks()
    print(f"\n[Data] Total combined static landmark samples: {len(X)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_cat, test_size=0.30, random_state=42, stratify=y_enc)
    y_tmp_enc = np.argmax(y_tmp, axis=1)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp_enc)
    
    model = build_landmark_mlp(num_classes)
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint(BEST_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    t0 = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, callbacks=cbs)
    print(f"\n[Train] Finished in {time.time()-t0:.0f}s")

    best = tf.keras.models.load_model(BEST_MODEL_PATH)
    loss, acc = best.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Eval] Test Accuracy: {acc*100:.2f}%")

    y_pred = np.argmax(best.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\n[Eval] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('MLP Confusion Matrix (Validation)')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    print(f"[Save] Confusion matrix saved → {CONFUSION_MATRIX_PATH}")

    best.save(FINAL_MODEL_PATH)
    np.save(LABEL_PATH, le.classes_)
    print(f"[Save] Saved model → {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
