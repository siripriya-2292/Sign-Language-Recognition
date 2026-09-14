import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# =====================================================
# SIGN NAMES
# =====================================================

SIGNS = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = load_model(
    "models/sign_language_model.keras",
    compile=False
)

# =====================================================
# LOAD SCALER
# =====================================================

scaler_mean = np.load("models/scaler_mean.npy")
scaler_scale = np.load("models/scaler_scale.npy")


# =====================================================
# MEDIAPIPE SETUP
# =====================================================

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


# =====================================================
# VARIABLES FOR WORD FORMATION
# =====================================================

word = ""

prediction_history = deque(maxlen=15)

last_added_sign = None

# Number of times the same sign must appear
STABLE_COUNT = 10


# =====================================================
# START CAMERA
# =====================================================

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert image
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)


        # =================================================
        # IF HAND IS DETECTED
        # =================================================

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


            # Scale data
            landmarks = (
                landmarks - scaler_mean
            ) / scaler_scale


            # Reshape
            landmarks = landmarks.reshape(1, 63)


            # Predict
            prediction = model.predict(
                landmarks,
                verbose=0
            )


            predicted_index = np.argmax(prediction)

            predicted_sign = SIGNS[predicted_index]

            confidence = prediction[0][predicted_index] * 100


            # =================================================
            # STORE PREDICTION
            # =================================================

            prediction_history.append(predicted_sign)


            # =================================================
            # CHECK STABLE PREDICTION
            # =================================================

            if len(prediction_history) == 15:

                most_common_sign = max(
                    set(prediction_history),
                    key=prediction_history.count
                )

                count = prediction_history.count(
                    most_common_sign
                )


                if count >= STABLE_COUNT:

                    # Add only if it is different
                    # from the previously added letter
                    if most_common_sign != last_added_sign:

                        word += most_common_sign

                        last_added_sign = most_common_sign

                        prediction_history.clear()


            # =================================================
            # DISPLAY SIGN
            # =================================================

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


            # =================================================
            # DISPLAY WORD
            # =================================================

            cv2.putText(
                frame,
                f"Text: {word}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )


            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


        # =================================================
        # NO HAND DETECTED
        # =================================================

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


        # =================================================
        # INSTRUCTIONS
        # =================================================

        cv2.putText(
            frame,
            "Q = Quit | C = Clear | SPACE = Space | B = Backspace",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # =================================================
        # SHOW CAMERA
        # =================================================

        cv2.imshow(
            "Sign Language Recognition",
            frame
        )


        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        # Quit
        if key == ord("q"):
            break


        # Clear word
        elif key == ord("c"):
            word = ""
            last_added_sign = None
            prediction_history.clear()


        # Space
        elif key == 32:
            word += " "
            last_added_sign = None
            prediction_history.clear()


        # Backspace
        elif key == ord("b"):
            word = word[:-1]
            last_added_sign = None
            prediction_history.clear()


    # =====================================================
    # RELEASE CAMERA
    # =====================================================

    cap.release()
    cv2.destroyAllWindows()

