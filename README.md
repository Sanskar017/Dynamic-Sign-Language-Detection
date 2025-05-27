# 🤟 Real-Time Dynamic Indian Sign Language Detection

A real-time Indian Sign Language (ISL) detection system using deep learning that recognizes both **dynamic phrases** and **static alphabets** from live webcam input. It leverages a hybrid **CNN-LSTM architecture** to process video data, enabling inclusive and efficient gesture recognition.

---

## 📖 About the Project

This project addresses the lack of real-time, dynamic ISL recognition systems. Most existing tools focus on American Sign Language (ASL) and static signs. Here, we introduce a vision-based solution capable of detecting full ISL phrases using live video without relying on gloves, sensors, or pre-trained Roboflow models.

The model architecture uses:
- **MobileNetV2 (CNN)** for spatial feature extraction.
- **LSTM** for capturing temporal dynamics across 30-frame gesture sequences.

The project shows strong performance on both static and dynamic data, achieving up to **98% accuracy** for alphabets and over **73% accuracy** for complex dynamic phrases like "What is your name?" and "Thank you".

---

## ✅ Features

- 🔴 Real-time gesture detection from webcam
- 📚 Detects full ISL words and sentences
- 🧠 Hybrid CNN-LSTM deep learning architecture
- 🧩 Custom preprocessing and augmentation pipeline
- 🔊 Optional text-to-speech integration

---

## 🧠 Model Architecture

The model handles both spatial and temporal features in sign language videos using a two-part deep learning pipeline:

### 🔧 Components

1. **TimeDistributed MobileNetV2 (CNN)**
   - Pre-trained on ImageNet (transfer learning)
   - Extracts spatial features from each frame (224x224 resolution)
   - TimeDistributed wrapper applies the same CNN to each of the 30 frames independently

2. **LSTM Layer**
   - 128 hidden units
   - Captures sequential motion patterns across the 30-frame buffer

3. **Dense Layers**
   - Fully connected layer with ReLU activation
   - Final Dense + Softmax layer to classify gestures into predefined sentence classes

### 🧱 Architecture Flow

├── preprocess.py # Prepares and saves x.npy, y.npy
├── train_model.py # CNN-LSTM training logic
├── realtime_inference.py # Live webcam detection
├── x2_filtered.npy/ # Processed x.npy file
├── y2_filtered.npy/ # Processed y.npy file
├── classes1.npy/ #  Processed classes.npy file
├── models/ # Trained model .h5 files
├── requirements.txt /# contains all the libraries 

## ⚙️ Installation

```bash
git clone https://github.com/Sanskar017/Dynamic-Sign-Language-Detection.git
cd Dynamic-Sign-Language-Detection
pip install -r requirements.txt
