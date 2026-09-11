import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Sign names
SIGNS = ["A", "B", "C", "D", "E"]

# Load trained model
model = load_model("models/sign_language_model.keras")

# Load scaler
scaler_mean = np.load("models/scaler_mean.npy")
scaler_scale = np.load("models/scaler_scale.npy")


# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        frame = cv2.flip(frame, 1)

        # Convert image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            landmarks = []

            for landmark in hand:

                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Convert to NumPy array
            landmarks = np.array(landmarks)

            # Scale the data
            landmarks = (
                landmarks - scaler_mean
            ) / scaler_scale

            # Reshape for model
            landmarks = landmarks.reshape(1, 63)

            # Predict
            prediction = model.predict(
                landmarks,
                verbose=0
            )

            predicted_index = np.argmax(prediction)

            predicted_sign = SIGNS[predicted_index]

            confidence = prediction[0][predicted_index] * 100

            # Display prediction
            cv2.putText(
                frame,
                f"Sign: {predicted_sign}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Draw hand landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        else:

            cv2.putText(
                frame,
                "Show your hand",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Sign Language Recognition",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()