# 🎥 SCDAS — Smart Camera-Based Detection & Analysis System

> A real-time and single-frame intelligent detection system powered by **PyTorch**, **OpenCV**, and **Librosa** — designed for visual and acoustic analysis through live camera feeds.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Flowchart](#flowchart)
- [Project Structure](#project-structure)
- [Modules Explained](#modules-explained)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Camera Test](#camera-test)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

**SCDAS (Smart Camera-Based Detection & Analysis System)** is an AI-powered computer vision and audio analysis system that operates in two modes:

- **Real-Time Mode** — Continuously captures live camera frames and performs detection/analysis on each frame in real time using a PyTorch-based deep learning model.
- **Single Frame Mode** — Captures or loads a single frame/image and performs a one-shot detection and analysis pass.

The system combines **visual detection** (via OpenCV and PyTorch) with **acoustic/audio analysis** (via Librosa and SciPy), making it suitable for multimodal intelligent sensing applications such as:

- Driver drowsiness detection
- Anomaly detection in surveillance
- Smart attendance systems
- Gesture or emotion recognition
- Audio-visual event detection

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 **Real-Time Detection** | Live camera feed processing with frame-by-frame AI inference |
| 🖼️ **Single Frame Analysis** | One-shot detection from a captured or loaded image |
| 🔊 **Audio Analysis** | Librosa-powered acoustic feature extraction (MFCCs, spectrograms) |
| 🤖 **Deep Learning Backend** | PyTorch model for classification/detection tasks |
| 📷 **Camera Diagnostics** | Built-in camera test utility to verify hardware connectivity |
| 🧮 **Signal Processing** | SciPy-based filtering and signal analysis |
| 🔢 **NumPy Pipelines** | Efficient numerical processing for frame and audio data |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SCDAS SYSTEM                               │
│                         Entry: main.py                              │
└─────────────────────┬───────────────────────┬───────────────────────┘
                      │                       │
         ┌────────────▼──────────┐ ┌──────────▼────────────┐
         │   Real-Time Demo      │ │  Single Frame Demo     │
         │  demo_realtime.py     │ │  demo_single_frame.py  │
         └────────────┬──────────┘ └──────────┬─────────────┘
                      │                       │
         ┌────────────▼───────────────────────▼─────────────┐
         │              INPUT LAYER                          │
         │                                                   │
         │   📷 Camera (cv2.VideoCapture)                    │
         │   🖼️  Image File / Single Frame                   │
         │   🔊 Audio Input (Librosa)                        │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │           PREPROCESSING LAYER                     │
         │                                                   │
         │   • Frame Resize & Normalization (OpenCV)         │
         │   • Color Space Conversion (BGR → RGB)            │
         │   • Audio Feature Extraction (MFCC, Mel, etc.)    │
         │   • Tensor Conversion (NumPy → PyTorch Tensor)    │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │           INFERENCE / MODEL LAYER                 │
         │                                                   │
         │   • PyTorch Neural Network (CNN / Hybrid)         │
         │   • Forward Pass → Predictions                    │
         │   • Confidence Scores & Class Labels              │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │           POSTPROCESSING LAYER                    │
         │                                                   │
         │   • Bounding Box / Landmark Drawing (OpenCV)      │
         │   • Label Overlay on Frames                       │
         │   • SciPy Signal Filtering on Audio Output        │
         │   • Result Formatting & Logging                   │
         └────────────────────────┬──────────────────────────┘
                                  │
         ┌────────────────────────▼──────────────────────────┐
         │              OUTPUT LAYER                         │
         │                                                   │
         │   📺 Live Display Window (cv2.imshow)             │
         │   📊 Detection Results (Console / Log)            │
         │   💾 Saved Output Frame / Report (optional)       │
         └───────────────────────────────────────────────────┘
```

---

## 🔄 Flowchart

### Main Program Flow

```
                    ┌─────────────────────┐
                    │     START SCDAS     │
                    │      main.py        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Display Menu       │
                    │  1. Real-Time Demo  │
                    │  2. Single Frame    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   User Input        │
                    │   Choice: 1 or 2    │
                    └──────┬──────┬───────┘
                          "1"    "2"
                           │      │
           ┌───────────────▼┐    ┌▼──────────────────┐
           │  demo_realtime │    │ demo_single_frame  │
           └───────┬────────┘    └────────┬───────────┘
                   │                      │
                   ▼                      ▼
    ┌──────────────────────┐   ┌──────────────────────────┐
    │  Open Camera         │   │  Load / Capture Image    │
    │  cv2.VideoCapture(0) │   │  Single Frame Input      │
    └──────────┬───────────┘   └────────────┬─────────────┘
               │                            │
    ┌──────────▼───────────┐                │
    │  Camera Opened?      │                │
    └────┬─────────┬───────┘                │
        YES        NO                       │
         │          │                       │
         │   ┌──────▼──────────┐            │
         │   │  Error: Exit    │            │
         │   └─────────────────┘            │
         │                                  │
    ┌────▼──────────────────────────────────▼──────┐
    │           PREPROCESSING                      │
    │  • Resize Frame                              │
    │  • Normalize Pixel Values                    │
    │  • Convert to PyTorch Tensor                 │
    │  • Extract Audio Features (if applicable)   │
    └────────────────────┬─────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────┐
    │           MODEL INFERENCE                    │
    │  • Load PyTorch Model                        │
    │  • Run Forward Pass                          │
    │  • Get Predictions & Confidence              │
    └────────────────────┬─────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────┐
    │           POSTPROCESSING                     │
    │  • Draw Bounding Boxes / Labels              │
    │  • Apply SciPy Filters (audio)               │
    │  • Format Output Results                     │
    └────────────────────┬─────────────────────────┘
                         │
    ┌────────────────────▼─────────────────────────┐
    │              DISPLAY OUTPUT                  │
    │  cv2.imshow() → Live Window                  │
    └────────────────────┬─────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Real-Time Mode?    │
              └──────┬──────┬───────┘
                    YES     NO
                     │       │
          ┌──────────▼──┐   ┌▼───────────────┐
          │  Press 'q'  │   │  Show Result   │
          │  to Quit?   │   │  and Exit      │
          └──────┬───┬──┘   └────────────────┘
                YES  NO
                 │    │
                 │    └──────────► Loop back to capture next frame
                 │
    ┌────────────▼──────────────┐
    │  Release Camera           │
    │  Destroy All Windows      │
    │  cap.release()            │
    │  cv2.destroyAllWindows()  │
    └────────────┬──────────────┘
                 │
    ┌────────────▼──────────────┐
    │          END              │
    └───────────────────────────┘
```

### Camera Test Flow (`test_camera.py`)

```
         ┌─────────────────────────────┐
         │  Run test_camera.py         │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  cv2.VideoCapture(0)        │
         │  Open Default Camera        │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  cap.isOpened()?            │
         └───────┬──────────┬──────────┘
                YES          NO
                 │            │
                 │   ┌────────▼──────────────┐
                 │   │ ❌ Camera not opened  │
                 │   │    Exit               │
                 │   └───────────────────────┘
                 │
         ┌───────▼─────────────────────┐
         │  ✅ Camera opened           │
         │  Start Frame Loop           │
         └───────────────┬─────────────┘
                         │
         ┌───────────────▼─────────────┐
         │  cap.read() → ret, frame    │
         └───────┬───────────┬─────────┘
               ret=True    ret=False
                 │            │
                 │   ┌────────▼────────────────┐
                 │   │ ❌ Failed to read frame │
                 │   │    Break Loop           │
                 │   └─────────────────────────┘
                 │
         ┌───────▼─────────────────────┐
         │  cv2.imshow("Camera Test")  │
         └───────────────┬─────────────┘
                         │
         ┌───────────────▼─────────────┐
         │  Key Press = 'q'?           │
         └──────┬───────────┬──────────┘
               YES           NO
                │             │
                │             └────► Loop back to cap.read()
                │
         ┌──────▼──────────────────────┐
         │  cap.release()              │
         │  cv2.destroyAllWindows()    │
         └─────────────────────────────┘
```

---

## 📁 Project Structure

```
SCDAS/
│
├── main.py                    # Entry point — mode selector
├── requirements.txt           # Python dependencies
├── test_camera.py             # Camera hardware diagnostic tool
│
├── demos/
│   ├── __init__.py
│   ├── demo_realtime.py       # Real-time camera detection demo
│   └── demo_single_frame.py   # Single frame detection demo
│
├── models/
│   ├── __init__.py
│   └── detector.py            # PyTorch model definition & loader
│
├── utils/
│   ├── __init__.py
│   ├── preprocess.py          # Frame & audio preprocessing utilities
│   ├── postprocess.py         # Result formatting & visualization
│   └── audio_analysis.py      # Librosa-based audio feature extraction
│
├── assets/
│   └── test_image.jpg         # Sample image for single frame demo
│
└── outputs/
    └── results/               # Saved detection outputs (optional)
```

---

## 🧩 Modules Explained

### `main.py` — Entry Point
The central launcher of the SCDAS system. Presents a CLI menu to the user and routes to either the real-time or single-frame demo based on input.

```python
from demos.demo_realtime import demo_realtime
from demos.demo_single_frame import demo_single_frame

choice = input("Enter choice: ")
if choice == '1':
    demo_realtime()       # Live camera feed
elif choice == '2':
    demo_single_frame()   # One-shot image analysis
```

---

### `demos/demo_realtime.py` — Real-Time Detection
- Opens live camera feed using `cv2.VideoCapture(0)`
- Reads frames in a continuous loop
- Preprocesses each frame and passes it through the PyTorch model
- Overlays detection results on the live feed using OpenCV drawing functions
- Displays using `cv2.imshow()` — press `q` to quit and release resources

---

### `demos/demo_single_frame.py` — Single Frame Detection
- Loads a single image or captures one frame from the camera
- Runs the full preprocessing → inference → postprocessing pipeline once
- Displays and/or saves the result
- Ideal for testing, debugging, or batch offline analysis

---

### `test_camera.py` — Camera Diagnostic Tool
A standalone utility to verify that the system's camera is correctly accessible before running the full SCDAS pipeline.

```python
cap = cv2.VideoCapture(0)       # Open default camera

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened successfully")
# Continuously display frames until 'q' is pressed
```

---

### PyTorch Model Layer
- Neural network (CNN or hybrid architecture) for visual/audio classification
- Loaded via `torch.load()` or model class instantiation
- Runs inference using `model.eval()` and `torch.no_grad()` for efficiency
- Outputs class predictions with confidence scores

---

### Audio Analysis (Librosa + SciPy)
- **Librosa** extracts audio features: MFCCs, Mel spectrograms, chroma features, zero-crossing rate
- **SciPy** applies signal filtering: bandpass, lowpass, noise reduction
- Audio features can be fused with visual features for multimodal detection

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| **PyTorch** | ≥ 2.0.0 | Deep learning inference engine |
| **OpenCV** | ≥ 4.8.0 | Camera capture, frame processing, display |
| **NumPy** | ≥ 1.24.0 | Numerical array operations and tensor prep |
| **Librosa** | ≥ 0.10.0 | Audio loading and feature extraction |
| **SciPy** | ≥ 1.11.0 | Signal processing and scientific computation |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam / Camera device connected
- (Optional) NVIDIA GPU with CUDA for faster PyTorch inference

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/SCDAS.git
cd SCDAS

# 2. Create a virtual environment (recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Optional: Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Run the Main System

```bash
python main.py
```

You will see:

```
SCDAS SYSTEM
1. Real-time Demo
2. Single Frame Demo
Enter choice:
```

- Enter `1` → **Real-Time Detection** (live camera feed, continuous inference)
- Enter `2` → **Single Frame Detection** (one-shot image analysis)

### Controls During Real-Time Mode

| Key | Action |
|---|---|
| `q` | Quit and exit the application |

---

## 📷 Camera Test

Before running the full system, verify your camera works:

```bash
python test_camera.py
```

**Expected Output:**
```
✅ Camera opened successfully
```

A live camera window will open. Press `q` to close it.

**If you see:**
```
❌ Camera not opened
```
Check that your webcam is properly connected, no other application is using the camera, and camera index `0` is correct (try `1` or `2` for external cameras).

---

## 📦 Requirements

All dependencies listed in `requirements.txt`:

```
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
librosa>=0.10.0
scipy>=1.11.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `❌ Camera not opened` | Camera unavailable or in use | Close other apps using camera; check device index |
| `❌ Failed to read frame` | Camera disconnected mid-session | Reconnect camera and restart |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| Slow inference | No GPU / CPU only | Install CUDA-enabled PyTorch |
| Librosa import error | Missing audio backend | Run `pip install soundfile` |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with 🎥 OpenCV · 🔥 PyTorch · 🔊 Librosa</p>
