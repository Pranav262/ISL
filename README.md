# 🖐️ SignBridge — Indian Sign Language (ISL) Recognition System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16.2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-00979D?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/edge/mediapipe/solutions/guide)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**SignBridge** is a full-stack, real-time **Indian Sign Language (ISL)** recognition platform. It features a Flask REST backend serving a fine-tuned MobileNetV2 Deep Neural Network, a live browser dashboard with 3D hand skeleton tracking, and a word-building interface for translating hand gestures into sentences in real time.

---

## 📂 Available Models & Current Production Setup

All trained model weights and class mappings are located under [`backend/model/`](file:///Users/pranav/Desktop/SignBridge/backend/model):

| Model File | Architecture | Size | Status in `app.py` | Benchmark Accuracy | Avg Latency |
|---|---|---|---|---|---|
| **`best_model.keras`** | **MobileNetV2 Transfer Learning CNN** | **23.8 MB** | **✅ CURRENTLY ACTIVE** | **87.3%** | **67.0 ms** |
| `signbridge_mobilenetv2.keras` | MobileNetV2 CNN (Final Checkpoint) | 23.8 MB | 📦 Backup | 87.3% | 67.0 ms |
| `signbridge_landmark_model.keras` | MediaPipe 21 Landmark MLP | 859 KB | 📦 Backup | 28.5% | 97.9 ms |
| `label_classes.npy` | Class Label Encoder (`a`–`z`, `{`) | < 1 KB | ✅ Active Mapping | — | — |

### 🎯 Why MobileNetV2 is Used in Production (`app.py`)

We conducted an empirical benchmark across **810 dataset test samples** evaluating both model architectures on accuracy, processing latency, and robustness.

1. **Superior Recognition Accuracy**: MobileNetV2 achieved **87.3% overall accuracy**, outperforming the MediaPipe Landmark DNN (**28.5%**). MobileNetV2 scored **100% accuracy** on 10 distinct gesture classes (`a`, `b`, `l`, `o`, `p`, `r`, `v`, `w`, `x`, `y`, `z`).
2. **Lower Real-Time Latency**: MobileNetV2 executes in **67.0 ms/frame** on M1 Apple Silicon compared to **97.9 ms/frame** for the landmark pipeline.
3. **No Detection Skips**: MediaPipe landmark detection failed to detect hands on 137 dataset images (16.9% failure rate on complex sign angles), whereas MobileNetV2 successfully processed 100% of incoming visual frames.
4. **Hybrid Engine**: In `app.py`, MobileNetV2 performs high-accuracy classification while MediaPipe runs in parallel to draw real-time 3D hand skeletons on the UI canvas.

---

## 📊 Dataset & Training Methodology

### 1. Dataset Overview
- **Classes**: 27 classes corresponding to ISL Alphabets `a` through `z` and `{` (Blank/Space gesture).
- **Format**: RGB images organized by folder per class.

### 2. MobileNetV2 Transfer Learning Pipeline (`notebooks/SuperBridge.ipynb`)
- **Preprocessing & Augmentation**:
  - Resized to `(224, 224, 3)` with MobileNetV2 `preprocess_input`.
  - Data Augmentation: `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.1)`, `RandomContrast(0.1)`.
- **Split**: 80% Training, 20% Validation (Validation further split 50/50 into Validation and Test sets).
- **Training Strategy**:
  - **Phase 1 (Frozen Base)**: Trained top classification head (GAP $\rightarrow$ Dropout 0.3 $\rightarrow$ Dense 128 $\rightarrow$ Dropout 0.2 $\rightarrow$ Softmax 27) for 20 epochs with Adam (`lr=1e-3`).
  - **Phase 2 (Fine-Tuning)**: Unfroze top 30 layers of MobileNetV2 base and trained for 10 epochs with Adam (`lr=1e-5`).

### 3. MediaPipe Landmark MLP Pipeline (`notebooks/SuperBridge_Landmark.ipynb`)
- **Feature Extraction**: Extracted 21 3D hand keypoints $(x, y, z)$ per hand using MediaPipe Hands, normalized relative to wrist origin and scaled by wrist-to-middle-finger distance.
- **Vector**: 126-dimensional normalized feature vector ($2 \text{ hands} \times 21 \text{ keypoints} \times 3 \text{ coordinates}$).
- **Architecture**: Multi-layer Dense Neural Network (Input 126 $\rightarrow$ Dense 256 $\rightarrow$ BatchNorm $\rightarrow$ Dropout 0.3 $\rightarrow$ Dense 128 $\rightarrow$ Softmax 27).

---

## 📈 Benchmark Evaluation & Comparison

Tested across 810 samples (30 images per class across 27 classes):

```text
============================================================
🏆  BENCHMARK SUMMARY
============================================================
  MediaPipe Landmark DNN : 28.5% accuracy  |  97.9 ms avg
  MobileNetV2 CNN        : 87.3% accuracy  |  67.0 ms avg

  → Best model for accuracy: MobileNetV2
  → Best model for speed:    MobileNetV2
============================================================
```

### Per-Class Accuracy Breakdown

| Class | MobileNetV2 CNN | MediaPipe Landmark DNN |
|:---:|:---:|:---:|
| **a** | **100.0%** | 6.9% |
| **b** | **100.0%** | 13.3% |
| **c** | 76.7% | **96.2%** |
| **d** | **96.7%** | 0.0% |
| **e** | **73.3%** | 3.3% |
| **f** | **56.7%** | 13.8% |
| **g** | **90.0%** | 16.0% |
| **h** | **83.3%** | 8.0% |
| **i** | **93.3%** | 78.6% |
| **j** | **60.0%** | 0.0% |
| **k** | **96.7%** | 56.0% |
| **l** | **100.0%** | 53.8% |
| **m** | **73.3%** | 36.0% |
| **n** | **23.3%** | 0.0% |
| **o** | **100.0%** | **100.0%** |
| **p** | **100.0%** | 53.6% |
| **q** | **86.7%** | 64.3% |
| **r** | **100.0%** | 0.0% |
| **s** | **96.7%** | 27.3% |
| **t** | **83.3%** | 20.0% |
| **u** | **90.0%** | 23.3% |
| **v** | **100.0%** | 0.0% |
| **w** | **100.0%** | 0.0% |
| **x** | **100.0%** | 0.0% |
| **y** | **100.0%** | 20.7% |
| **z** | **100.0%** | 72.7% |
| **{ (space)** | **76.7%** | — |

---

## 📁 Project Directory Structure

```text
SignBridge/
├── README.md                         # Detailed project documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusions
├── benchmark_models.py               # Empirical evaluation & benchmarking script
│
├── backend/
│   ├── app.py                        # Flask REST API server (MobileNetV2 + MediaPipe)
│   └── model/
│       ├── best_model.keras                 # ✅ Active MobileNetV2 CNN (23.8 MB)
│       ├── signbridge_mobilenetv2.keras     # 📦 MobileNetV2 Final Checkpoint (23.8 MB)
│       ├── signbridge_landmark_model.keras  # 📦 MediaPipe Landmark DNN Backup (859 KB)
│       └── label_classes.npy               # Saved class label array
│
├── frontend/
│   ├── index.html                    # Dashboard UI layout
│   ├── style.css                     # Premium dark glassmorphism styling
│   └── app.js                        # Webcam capture, API polling, canvas overlay
│
└── notebooks/
    ├── SuperBridge.ipynb             # MobileNetV2 model training notebook
    └── SuperBridge_Landmark.ipynb    # MediaPipe landmark extraction & DNN notebook
```

---

## 🚀 Local Installation & Execution

### 1. Setup Virtual Environment
```bash
git clone https://github.com/Pranav262/SignBridge.git
cd SignBridge

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the Backend API
```bash
source .venv/bin/activate
python backend/app.py
```

### 3. Launch Live Dashboard
Open your browser and navigate to:
**`http://localhost:5050`**

---

## 💻 Dashboard Features

- 🎥 **Real-time Webcam Stream**: Low-latency video capture via HTML5 `getUserMedia`.
- 🦴 **MediaPipe Skeleton Overlay**: 21-point 3D hand keypoint tracking drawn dynamically on canvas.
- 🔤 **Live Letter Prediction**: High-contrast, animated letter display with confidence glow.
- 🎯 **Confidence Ring & Top-5 Probabilities**: Live visual breakdown of model prediction confidence.
- 📝 **Word Builder & Sentence Formatter**: Auto-commits letters after 2 stable frames into words and sentences.
- 📋 **Log History**: Timestamped detection log.

---

## 📜 License & Credits

Developed by **[Pranav262](https://github.com/Pranav262)**. Powered by [TensorFlow](https://www.tensorflow.org/), [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide), [Flask](https://flask.palletsprojects.com/), and [OpenCV](https://opencv.org/).
