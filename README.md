# 🤟 Sign Language Recognition

A real-time **Sign Language Recognition system** that uses **Computer Vision and Deep Learning** to recognize American Sign Language (ASL) alphabet gestures from a webcam.

The system detects the hand using **MediaPipe**, extracts hand landmarks, and uses a trained **TensorFlow/Keras neural network** to predict the corresponding alphabet from **A to Z**.

---

## 📌 Project Overview

This project recognizes hand gestures representing English alphabets **A–Z** using a webcam.

The system works in the following steps:

1. Capture the user's hand through the webcam.
2. Detect the hand using MediaPipe.
3. Extract **21 hand landmarks**.
4. Convert the landmarks into numerical features.
5. Normalize the features using a saved scaler.
6. Pass the features to the trained neural network.
7. Predict the corresponding alphabet.
8. Display the predicted sign and confidence on the screen.
9. Hold a sign for about **1 second** to add the letter to the recognized text.

---

## 🎯 Objective

The main objective of this project is to develop a simple and real-time system that can recognize sign language alphabet gestures and convert them into readable text.

This project demonstrates the use of:

* Computer Vision
* Hand Landmark Detection
* Deep Learning
* Neural Networks
* Real-Time Webcam Processing

---

## 🛠️ Technologies Used

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| Python             | Main programming language   |
| TensorFlow / Keras | Deep Learning model         |
| MediaPipe          | Hand landmark detection     |
| OpenCV             | Image and webcam processing |
| NumPy              | Numerical operations        |
| Streamlit          | Web application interface   |
| Streamlit-WebRTC   | Real-time webcam streaming  |

---

## 🧠 Model

The trained model is a fully connected neural network designed to classify hand landmark features.

### Input

* 21 hand landmarks
* 3 coordinates per landmark: **X, Y, Z**
* Total input features: **63**

### Network Architecture

```text
63 Input Features
       ↓
Dense Layer (128 neurons)
       ↓
Dropout
       ↓
Dense Layer (64 neurons)
       ↓
Dropout
       ↓
Dense Layer (32 neurons)
       ↓
Dense Layer (26 neurons)
       ↓
Softmax
       ↓
A–Z Prediction
```

The final layer contains **26 output classes**, one for each English alphabet letter.

---

## 📊 Model Performance

The trained model achieved approximately:

**Test Accuracy: 94.42%**

The model was evaluated using classification metrics including precision, recall, and F1-score for the individual alphabet classes.

---

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── app.py
├── check_dataset.py
├── collect_data.py
├── hand_detection.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── models/
    ├── hand_landmarker.task
    ├── scaler_mean.npy
    ├── scaler_scale.npy
    └── sign_language_model.keras
```

---

## 📄 File Description

### `collect_data.py`

Used to collect hand landmark data for the alphabet classes.

### `check_dataset.py`

Used to check and verify the collected dataset.

### `hand_detection.py`

Detects the hand and extracts hand landmarks using MediaPipe.

### `train_model.py`

Prepares the dataset and trains the neural network model.

### `predict.py`

Uses the trained model to recognize signs through the webcam.

### `app.py`

Provides the Streamlit web interface with real-time webcam-based sign recognition.

### `models/`

Contains the trained neural network, scaler files, and MediaPipe hand landmark model.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/siripriya-2292/Sign-Language-Recognition.git
```

### 2. Open the project folder

```bash
cd Sign-Language-Recognition
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🎥 How to Use

1. Start the Streamlit application.
2. Allow access to your webcam.
3. Show one alphabet sign in front of the camera.
4. Hold the sign for approximately **1 second**.
5. The predicted letter will appear on the camera.
6. Remove your hand.
7. Show the next letter.
8. Repeat to form a word.

### Example

```text
Show H → Remove hand
        ↓
Show E → Remove hand
        ↓
Show L → Remove hand
        ↓
Show L → Remove hand
        ↓
Show O

Result: HELLO
```

---

## ✨ Features

* 🤟 Real-time hand detection
* 🔤 A–Z alphabet recognition
* 📷 Webcam-based prediction
* 🧠 Deep Learning classification
* ✋ MediaPipe hand landmark detection
* 📊 Prediction confidence
* 📝 Real-time text formation
* 🌐 Streamlit web interface

---

## 🔮 Future Improvements

* Add recognition for complete words and sentences.
* Improve recognition of difficult alphabet classes.
* Add support for dynamic signs such as **J** and **Z**.
* Add voice output for recognized text.
* Add more training data for better generalization.
* Deploy the application online for public access.

---

## 👩‍💻 Author

**Siripriya**

Computer Science and Engineering Student

GitHub:
https://github.com/siripriya-2292

---

## ⭐ Acknowledgement

This project was developed as a learning project to explore **Deep Learning, Computer Vision, Hand Landmark Detection, and Real-Time Sign Language Recognition**.
