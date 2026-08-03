import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ---- 1. Load Pretrained Landmark Model & Class Labels ----
MODEL_PATHS = ["./best_landmark_model.keras", "./artifacts/best_landmark_model.keras"]
LABEL_PATHS = ["./label_classes.npy", "./artifacts/label_classes.npy"]

model_path = next((p for p in MODEL_PATHS if os.path.exists(p)), None)
label_path = next((p for p in LABEL_PATHS if os.path.exists(p)), None)

if not model_path or not label_path:
    raise FileNotFoundError("Could not locate landmark model or label_classes.npy artifact.")

print(f"Loading model from: {model_path}")
print(f"Loading labels from: {label_path}")

model = tf.keras.models.load_model(model_path)
label_classes = np.load(label_path, allow_pickle=True)
print(f"Loaded {len(label_classes)} classes: {list(label_classes)}")

# ---- 2. MediaPipe Setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_two_hand_landmarks(img_rgb, hands_detector):
    """Extracts normalized 126-dim vector (Left hand + Right hand)."""
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks or not res.multi_handedness:
        return None, None

    hands_dict = {}
    for lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handedness.classification[0].label  # "Left" or "Right"
        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)

        # Normalize: center on wrist, scale by wrist-to-middle-MCP distance
        wrist = coords[0]
        coords = coords - wrist
        scale = np.linalg.norm(coords[9]) + 1e-6
        coords = coords / scale

        hands_dict[label] = coords.flatten()

    left = hands_dict.get("Left", np.zeros(63, dtype=np.float32))
    right = hands_dict.get("Right", np.zeros(63, dtype=np.float32))
    feat = np.concatenate([left, right])
    return feat, res.multi_hand_landmarks

# ---- 3. Live Webcam Loop ----
def main():
    CONFIDENCE_THRESHOLD = 50.0  # % confidence threshold
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    print("\n" + "="*50)
    print(" 🚀 SignBridge Live Camera Real-Time Recognition")
    print(" Press 'q' or 'ESC' in the video window to exit.")
    print("="*50 + "\n")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera.")
                break

            # Mirror frame horizontally for intuitive self-view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            feat, landmarks_list = extract_two_hand_landmarks(frame_rgb, hands)

            display_text = "No Hand Detected"
            conf_score = 0.0

            if feat is not None:
                # Draw skeleton landmarks on video frame
                for hand_landmarks in landmarks_list:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Predict gesture
                preds = model.predict(feat.reshape(1, -1), verbose=0)[0]
                idx = np.argmax(preds)
                conf_score = float(preds[idx]) * 100
                raw_label = str(label_classes[idx])

                if conf_score >= CONFIDENCE_THRESHOLD:
                    display_text = "SPACE" if raw_label == "{" else raw_label.upper()
                else:
                    display_text = f"Uncertain ({raw_label.upper()}?)"

            # ---- Sleek UI Overlay ----
            cv2.rectangle(frame, (10, 10), (450, 100), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (450, 100), (0, 255, 200), 2)

            color = (0, 255, 0) if conf_score >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            if feat is None:
                color = (180, 180, 180)

            cv2.putText(frame, f"SIGN: {display_text}", (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {conf_score:.1f}%", (25, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("SignBridge - Real-Time ISL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time recognition session ended.")

if __name__ == "__main__":
    main()
