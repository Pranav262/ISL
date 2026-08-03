# 🖐️ SignBridge — Indian Sign Language (ISL) Recognition

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00979D?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**SignBridge** is a Deep Learning and Computer Vision framework designed for automated **Indian Sign Language (ISL)** alphabet and gesture recognition (classes `a` through `z` plus blank/space gestures). 

The repository provides two complementary training pipelines:
1. **End-to-End Image Classification**: MobileNetV2 transfer learning on full RGB gesture images.
2. **Real-Time Hand Landmark Extraction**: MediaPipe 21 3D hand keypoint coordinate extraction paired with a lightweight Deep Neural Network (DNN) for low-latency, real-time edge inference.

---

## 📁 Project Structure

```text
SignBridge/
├── README.md                      # Project documentation
├── requirements.txt               # Exact Python dependencies
├── .gitignore                     # Git exclusions (.venv, dataset, cache)
│
├── 📓 Notebooks
│   ├── SuperBridge.ipynb          # Pipeline 1: MobileNetV2 Transfer Learning on raw images
│   └── SuperBridge_Landmark.ipynb # Pipeline 2: MediaPipe 21 Hand Landmark Feature Extraction & DNN
│
├── 🧠 Pretrained Model Checkpoints
│   ├── best_model.keras           # Top-performing MobileNetV2 weights (Validation Accuracy)
│   ├── signbridge_mobilenetv2.keras # Final saved MobileNetV2 model
│   ├── best_landmark_model.keras  # Top-performing MediaPipe Landmark DNN model
│   └── signbridge_landmark_model.keras # Final saved Landmark DNN model
│
├── 📊 Extracted Datasets & Artifacts
│   ├── landmark_dataset.npz       # Pre-extracted 3D hand keypoint feature dataset
│   ├── label_classes.npy          # Saved class label mapping (a-z, space)
│   └── artifacts/                 # Saved model checkpoints and numpy artifacts
│
└── 🖼️ Visualizations & Assets
    ├── image.png                  # Sample gesture / training visualization
    ├── image2.png                 # Model evaluation / confusion matrix plot
    └── image3.jpg                 # Single-image inference demonstration
```

---

## ⚡ Technical Architectures

### 1. MobileNetV2 Transfer Learning (`SuperBridge.ipynb`)
- **Input**: `(224, 224, 3)` RGB Sign Images.
- **Base Architecture**: Pretrained `MobileNetV2` on ImageNet (Frozen base $\rightarrow$ Top ~30 layers un-frozen for fine-tuning).
- **Head**: Global Average Pooling $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (128, ReLU) $\rightarrow$ Dropout (0.2) $\rightarrow$ Softmax output ($N$ classes).
- **Optimization**: Adam optimizer with initial `lr=1e-3` (base) and `lr=1e-5` (fine-tuning), coupled with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.

### 2. MediaPipe Hand Landmark DNN (`SuperBridge_Landmark.ipynb`)
- **Input**: 63-dimensional normalized vector representing $(x, y, z)$ spatial coordinates for 21 3D hand landmarks detected via Google MediaPipe Hand Landmarker.
- **Model**: Fully Connected Deep Neural Network (Dense layers + Batch Normalization + Dropout) designed for zero-latency real-time video stream execution on low-power devices.

---

## 🛠️ Environment Setup & Installation

### 1. Prerequisites
- **Python 3.12** installed on your system.
- Recommended OS: **macOS (Apple Silicon M1/M2/M3/M4)**, Linux, or Windows.

### 2. Clone the Repository
```bash
git clone https://github.com/Pranav262/ISL.git
cd ISL
```

### 3. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Apple Silicon Mac GPU Acceleration (M1/M2/M3/M4)**:
> To enable Metal GPU acceleration on Mac:
> ```bash
> pip install tensorflow-metal
> ```

---

## 🚀 How to Run

### Option A: Launching Jupyter Notebook
```bash
jupyter notebook
```
- Open `SuperBridge.ipynb` to train/evaluate the **MobileNetV2** image model.
- Open `SuperBridge_Landmark.ipynb` to run **MediaPipe Landmark** extraction and DNN training.

### Option B: Quick Single-Image Inference Test
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model and class names
model = tf.keras.models.load_model("./best_model.keras")
class_names = list(np.load("./artifacts/label_classes.npy"))

def predict_sign(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    label = class_names[idx]
    display_label = "SPACE/BLANK" if label == "{" else label.upper()
    return {"alphabet": display_label, "confidence": f"{confidence:.2f}%"}

# Test sample image
print(predict_sign("./image3.jpg"))
```

---

## 📊 Dataset Structure

Organize your dataset folder in the following structure before running training cells:

```text
dataset/
├── a/  (images for gesture A)
├── b/  (images for gesture B)
...
├── z/  (images for gesture Z)
└── {/  (images for blank/space gesture)
```

---

## 📜 License & Credits

- Created & Maintained by **[Pranav262](https://github.com/Pranav262)**.
- Built using [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide), and [OpenCV](https://opencv.org/).
