# Multimodal physics-informed bearing fault diagnosis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bearing-dagnosis.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Group 10 – IIT Mandi & HCL Tech Hackathon**

Welcome to our solution for the **HCL Tech Hackathon** at **IIT Mandi**. This repository contains a complete pipeline for multimodal, physics-informed bearing fault diagnosis, from data preprocessing and model training to an interactive web application.

---

## 🌐 Live Demo

Experience the application instantly without any setup:

👉 **[bearing-dagnosis.streamlit.app](https://bearing-dagnosis.streamlit.app/)**

The live demo allows you to:
- **Upload your own `.mat` files** or use pre-loaded samples.
- **Interactively select signal windows** for precise diagnosis.
- **Explore order-domain spectrograms** and physics-based explanations.
- **Visualize model uncertainty** and branch attention weights.

---

## 🧠 Model Architecture

Our system employs a **multimodal, physics-informed neural network** designed for robust and interpretable fault diagnosis. It consists of three parallel branches that process different aspects of the vibration signal, fused by an attention mechanism.

### 1. Time Branch (1D CNN)
- **Purpose:** Captures temporal patterns and transients in raw vibration signals.
- **Architecture:** Two-scale 1D ResNet-like CNN.
    - **Short-scale:** Processes the first 1024 samples (focus on high-frequency impulses).
    - **Long-scale:** Processes the full 2048-sample window (captures periodic patterns).
- **Output:** A 256-dimensional feature vector.

### 2. Frequency Branch (2D CNN)
- **Purpose:** Analyzes spectral content in a machine-invariant manner.
- **Key Innovation: Order-Domain Processing**
    - Converts frequency (Hz) to **orders** (multiples of shaft frequency).
    - Fault signatures (e.g., BPFO at 5.43× shaft) appear at the *same* order regardless of RPM.
    - Makes the model robust to speed variations and different machines.
- **Architecture:** 2D CNN processing a 64×64 order-domain spectrogram.
- **Output:** A 256-dimensional feature vector.

### 3. Physics Branch (MLP)
- **Purpose:** Injects domain knowledge and verifies physical consistency.
- **Input:** A 30-dimensional feature vector computed directly from the signal:
    - FTF, BPF, BPFO, BPFI harmonics (1× to 4×).
    - BPFO/BPFI sidebands.
    - Statistical features: RMS, peak, crest factor, kurtosis, skewness, peak-to-peak.
    - Order band energies (0–2×, 2–5×, 5–10×, 10–20× shaft frequency).
- **Architecture:** An MLP that processes these physics-based features.
- **Output:** A 256-dimensional feature vector.

### 4. Attention Fusion
- **Purpose:** Dynamically weight the contribution of each branch for every input.
- **Mechanism:** Learns which modality (time, frequency, or physics) is most reliable for the given sample.
- **Output:** A fused 256-dimensional representation passed to the classifier and autoencoder.

### 5. Classifier & Autoencoder
- **Classifier:** A simple MLP that maps the fused features to one of four classes: `Healthy`, `Inner Race Fault`, `Ball Fault`, `Outer Race Fault`.
- **Autoencoder:** Trained exclusively on healthy data.
    - **Purpose:** Detects anomalies (unknown fault patterns) by measuring reconstruction error.
    - **Output:** A normalized anomaly score.

### 6. Uncertainty Quantification (MC Dropout)
- **Purpose:** Provides a measure of prediction confidence.
- **Mechanism:** Runs 15–30 stochastic forward passes with dropout enabled during inference.
- **Output:** The standard deviation of the predicted class probabilities.

---

## 📊 Performance

| Dataset | Accuracy | Macro F1 |
| :--- | :--- | :--- |
| **Internal (folders 1–7)** | 95.1% | 0.74 |
| **Unseen Machines (folders 8–10)** | 63.7% | 0.42 |

The model demonstrates strong generalization to completely unseen machines, significantly outperforming a random baseline (25%).

---

## 🚀 Quick Start

Get the app running locally in a few commands:

```bash
# 1. Clone the repository
git clone https://github.com/vika-spec/Bearing-Dagnosis.git
cd Bearing-Dagnosis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
