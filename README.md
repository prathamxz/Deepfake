# 🛡️ DeepGuard AI — Deepfake Detection System

A real-time deepfake detection system built with **EfficientNetV2B0 + CBAM Attention**, deployed as both a **Chrome Extension** and a **Streamlit web app** for AI-generated content detection — directly in the browser, with zero data leaving your device.

**Final Model Performance:**
| Metric | Score |
|---|---|
| Accuracy | **88.28%** |
| AUC-ROC | **0.9571** |
| Average Precision | **0.9585** |
| F1 Score (optimized threshold) | **0.8880** |

*Evaluated on a balanced test set of 20,000 samples (10,000 real, 10,000 fake), trained on ~140K real/fake face images.*

---

## 🌐 Chrome Extension

### Features
- **Real-time page scanning** — Automatically detects AI-generated images and videos on any web page
- **100% local inference** — All analysis runs on-device using TensorFlow.js; no data ever leaves your machine
- **Smart detection** — MutationObserver-based scanning catches dynamically loaded content (infinite scroll, SPAs)
- **Video frame analysis** — Extracts and analyzes video frames at configurable intervals
- **Premium UI** — Glassmorphism popup dashboard with real-time stats, detection history, and configurable settings
- **In-page overlays** — Non-intrusive badges on scanned images showing Real/AI-Generated status with confidence scores
- **User-configurable** — Toggle auto-scan, adjust sensitivity, show/hide overlays, enable/disable video scanning

### Extension Setup
1. **Convert the trained model** to TensorFlow.js format:
   ```bash
   pip install tensorflowjs
   python scripts/convert_model.py
   ```
   This produces `extension/model/model.json` + weight `.bin` files.

2. **Load the extension** in Chrome:
   - Open `chrome://extensions/`
   - Enable **Developer mode** (top-right toggle)
   - Click **Load unpacked**
   - Select the `extension/` folder

3. **Usage**:
   - Click the 🛡️ DeepGuard AI icon in the toolbar
   - Click **"Scan This Page"** to analyze all images/videos
   - Toggle **Auto-Scan** in settings to scan pages automatically
   - Hover over badges on images to see detailed confidence scores

### Extension Architecture
```
extension/
├── manifest.json          # Manifest V3 configuration
├── background.js          # Service worker (message routing, state management)
├── content/
│   ├── content.js         # DOM scanner, MutationObserver, overlay injection
│   └── overlay.css        # In-page glassmorphism badges
├── offscreen/
│   ├── offscreen.html     # Hidden document for TF.js inference
│   └── offscreen.js       # Model loading, preprocessing, prediction
├── popup/
│   ├── popup.html         # Dashboard UI
│   ├── popup.css          # Premium glassmorphism styles
│   └── popup.js           # Settings, stats, scan control
├── lib/
│   └── tf.min.js          # TensorFlow.js library (~1.4 MB)
├── model/
│   ├── model.json         # Converted model topology
│   └── *.bin              # Weight shards (~10-15 MB)
└── assets/
    └── icons/             # Extension icons (16, 48, 128px)
```

---

## 💻 Streamlit Web App

For deeper, manual analysis beyond the extension's automatic scanning. Powered by the same detection engine.

### Modes
- **Image Analysis** — Upload a single image and get an instant Real/AI-Generated verdict with confidence score
- **Video Analysis** — Upload a video for full per-frame breakdown, aggregated prediction, and fake-frame percentage
- **Live Capture** — Use your webcam to capture a photo and get instant on-the-spot deepfake verification

Each mode returns face detection results, confidence scoring, and a downloadable result summary/card.

### Launch
```bash
streamlit run app.py
```

---

## 🔬 ML Training Pipeline (Python)

### Folder Structure
- `data/raw`: Original dataset images/videos (managed by `download_data.py`)
- `data/processed`: Cropped faces and preprocessed data
- `models/`: Saved model weights (`.h5`, `.keras`)
- `src/`: Source code for preprocessing, training, and inference
- `notebooks/`: Jupyter notebooks for experimentation
- `scripts/`: Model conversion utilities
- `download_data.py`: Script to download and set up the dataset

### Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the environment:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup
1. **Kaggle API**:
   - Go to [Kaggle Account](https://www.kaggle.com/account)
   - Click "Create New API Token" to download `kaggle.json`
   - Place `kaggle.json` in `C:\Users\<YourUser>\.kaggle\` OR in this project folder
2. **Download Data**:
   ```bash
   python download_data.py
   ```
   This downloads the dataset and organizes it into `data/raw/real` and `data/raw/fake`.

### Data Preprocessing
Extract faces from videos/images using MTCNN:
```bash
python src/preprocess.py
```
This populates `data/processed/real` and `data/processed/fake`.

### Model Training
Train the EfficientNetV2B0 + CBAM model with two-phase training (frozen backbone → fine-tuning):
```bash
python src/train.py
```
This saves the model to `models/deepfake_detector.keras`.

### Evaluation
Generate a comprehensive evaluation report (confusion matrix, ROC curve, PR curve, threshold analysis):
```bash
python src/evaluate.py
```

---

## 🧠 Model Architecture

- **Base Model**: EfficientNetV2B0 (pre-trained on ImageNet)
- **Preprocessing**: MTCNN for face detection and extraction
- **Attention**: CBAM (Convolutional Block Attention Module) — focuses on facial regions like eyes, skin texture, and lighting seams
- **Custom Head**: GAP → Dense(512) → BN → Dropout → Dense(256) → BN → Dropout → Sigmoid
- **Training Strategy**: Two-phase (Phase 1: frozen backbone, classification head only → Phase 2: fine-tuning top layers)
- **Loss Function**: Binary Focal Crossentropy
- **Metrics Tracked**: Accuracy, AUC-ROC, Precision, Recall, F1-Score
- **Deployment Optimization**: Float16 quantization for TensorFlow.js, enabling fast browser-based inference via WebGL

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Test Set Size | 20,000 samples (balanced) |
| Accuracy | 88.28% |
| AUC-ROC | 0.9571 |
| Average Precision | 0.9585 |
| F1 @ 0.5 threshold | 0.8782 |
| Best F1 (optimized threshold = 0.438) | 0.8880 |
| True Positives / True Negatives | 8,450 / 9,206 |
| False Positives / False Negatives | 794 / 1,550 |

---

## 📄 Research Paper

This project has been submitted to **ICIAEM-2026** for academic review.

---

## Requirements

- Python 3.10+
- TensorFlow 2.12+
- Chrome 120+ (for extension)
- See `requirements.txt` for full dependency list

---

## 🔮 Future Scope

- Temporal (RNN/LSTM) modeling for video-native deepfake detection
- Larger, more diverse training datasets for improved generalization
- Cross-browser and mobile support
- Explainable AI features to visualize *why* content is flagged as fake

---

## Author

Built by **Prathamesh Zagade** ([@prathamxz](https://github.com/prathamxz)) as a final year B.E. project (Artificial Intelligence & Data Science).
