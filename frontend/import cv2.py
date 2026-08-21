import cv2
import numpy as np
import os
import mediapipe as mp

# -------------------- MediaPipe Setup --------------------
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# -------------------- Configuration --------------------
CLASSES = ["H", "J", "Y"]      # Add more signs if needed
NUM_PEOPLE = 5                 # Total number of participants
SAMPLES_PER_CLASS = 30         # Sequences per person per sign
SEQUENCE_LENGTH = 30           # Frames per sequence
DATA_DIR = "dataset_dynamic"

# -------------------- Extract 126 Keypoints --------------------
def extract_126_keypoints(results):
    left = np.zeros(63)
    right = np.zeros(63)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hl in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label

            pts = []
            for lm in hl.landmark:
                pts.extend([lm.x, lm.y, lm.z])

            if label == "Left":
                left = np.array(pts)
            else:
                right = np.array(pts)

    return np.concatenate([left, right])

# -------------------- Open/Scan Camera --------------------
def get_working_camera():
    print("\nScanning for active, non-blank camera...")
    # Typically 1 is the built-in webcam on Mac when Continuity Camera (0) is disconnected/blank.
    for index in [1, 0, 2, 3]:
        print(f"Testing camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Wait briefly for sensor to initialize
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    if np.sum(frame) > 100:  # Check that it's not entirely black
                        print(f"✅ Found working camera at index {index}!")
                        cap.release()
                        return index
            cap.release()
    print("⚠️ No non-blank cameras detected. Defaulting to index 0.")
    return 0

CAMERA_INDEX = get_working_camera()
cap = cv2.VideoCapture(CAMERA_INDEX)

def read_frame_safe(cap, camera_idx):
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("⚠️ Camera frame empty or disconnected. Attempting to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(camera_idx)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            # Create a black placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "CAMERA DISCONNECTED! Reconnecting...",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return cap, frame, False
    return cap, frame, True

# =====================================================
# Main Loop
# =====================================================
for label in CLASSES:

    print("\n===================================================")
    print(f"NOW COLLECTING SIGN: {label}")
    print("===================================================")

    # One sign, all 5 people
    for person in range(1, NUM_PEOPLE + 1):

        person_name = f"Person_{person}"
        save_dir = os.path.join(DATA_DIR, label, person_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n--------------------------------------")
        print(f"{person_name} - Perform Sign '{label}'")
        print("Press 'S' to start recording")
        print("--------------------------------------")

        # Wait until current person is ready
        while True:
            cap, frame, is_valid = read_frame_safe(cap, CAMERA_INDEX)
            if is_valid:
                frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                f"{label} | {person_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Press 'S' to Start" if is_valid else "Camera Offline - Reconnecting...",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255) if is_valid else (0, 0, 255),
                2,
            )

            cv2.imshow("Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and is_valid:
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Record samples
        for sample_num in range(SAMPLES_PER_CLASS):

            sequence_data = []

            # 3-second countdown (renders frame live rather than freezing)
            for countdown in range(3, 0, -1):
                start_time = cv2.getTickCount()
                frequency = cv2.getTickFrequency()
                while (cv2.getTickCount() - start_time) / frequency < 1.0:
                    cap, frame, is_valid = read_frame_safe(cap, CAMERA_INDEX)
                    if is_valid:
                        frame = cv2.flip(frame, 1)

                    cv2.putText(
                        frame,
                        f"{person_name} | {label}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Starting in {countdown}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )

                    cv2.imshow("Data Collector", frame)
                    if cv2.waitKey(20) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

            # Record sequence
            while len(sequence_data) < SEQUENCE_LENGTH:

                cap, frame, is_valid = read_frame_safe(cap, CAMERA_INDEX)
                if not is_valid:
                    cv2.imshow("Data Collector", frame)
                    cv2.waitKey(100)
                    continue

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand,
                            mp.solutions.hands.HAND_CONNECTIONS
                        )

                keypoints = extract_126_keypoints(results)
                sequence_data.append(keypoints)

                cv2.putText(
                    frame,
                    f"Sign : {label}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{person_name}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Sample : {sample_num+1}/{SAMPLES_PER_CLASS}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Frame : {len(sequence_data)}/{SEQUENCE_LENGTH}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("Data Collector", frame)

                if cv2.waitKey(20) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Save sample
            np.save(
                os.path.join(save_dir, f"sample_{sample_num}.npy"),
                np.array(sequence_data, dtype=np.float32)
            )

            print(
                f"{label} | {person_name} | "
                f"Sample {sample_num+1}/{SAMPLES_PER_CLASS} Saved"
            )

        print(f"\n✅ {person_name} completed sign '{label}'.")

print("\n===================================")
print("Dataset collection completed!")
print("===================================")

cap.release()
cv2.destroyAllWindows()