# Real-Time Sign Language Recognition Using Deep Learning

## 📌 Project Overview

This project is a Deep Learning based Sign Language Recognition System that recognizes hand signs using a webcam.

The system uses **MediaPipe** to detect hand landmarks and a **Neural Network** built with **TensorFlow** to classify the detected hand gesture.

Currently, the model recognizes five signs:

- A
- B
- C
- D
- E

## 🔄 How the System Works

```text
Webcam
   ↓
Hand Detection
   ↓
MediaPipe Hand Landmarks
   ↓
63 Landmark Features
   ↓
Neural Network
   ↓
Sign Prediction
   ↓
A / B / C / D / E
