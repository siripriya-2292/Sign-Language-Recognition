import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading
import time

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write("Show an alphabet sign and keep your hand steady.")

# ============================================================
# SIGN NAMES
# ============================================================

SIGNS = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_resources():

    model = load_model(
        "models/sign_language_model.keras",
        compile=False
    )

    scaler_mean = np.load(
        "models/scaler_mean.npy"
    )

    scaler_scale = np.load(
        "models/scaler_scale.npy"
    )

    return model, scaler_mean, scaler_scale


model, scaler_mean, scaler_scale = load_resources()

# ============================================================
# MEDIAPIPE
# ============================================================

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


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignLanguageProcessor(VideoProcessorBase):

    def __init__(self):

        self.current_sign = "No hand"
        self.confidence = 0.0

        # Final recognized text
        self.word = ""

        # Current sign being held
        self.last_sign = ""

        # Time when current sign started
        self.sign_start_time = None

        # Prevent adding same sign continuously
        self.letter_added = False

        # Thread lock
        self.lock = threading.Lock()

        # MediaPipe hand detector
        self.landmarker = HandLandmarker.create_from_options(
            options
        )

    def recv(self, frame):

        # ----------------------------------------------------
        # Get camera frame
        # ----------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        img = cv2.flip(img, 1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        result = self.landmarker.detect(
            mp_image
        )

        detected_sign = "No hand"
        detected_confidence = 0.0

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # ------------------------------------------------
            # Get 63 landmarks
            # ------------------------------------------------

            landmarks = []

            for landmark in hand:

                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            landmarks = np.array(
                landmarks,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Apply scaler
            # ------------------------------------------------

            landmarks = (
                landmarks - scaler_mean
            ) / scaler_scale

            landmarks = landmarks.reshape(
                1,
                63
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            prediction = model.predict(
                landmarks,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            detected_sign = SIGNS[
                predicted_index
            ]

            detected_confidence = (
                float(
                    prediction[0][predicted_index]
                ) * 100
            )

            # =================================================
            # WORD FORMATION
            # =================================================

            if detected_confidence >= 70:

                # First time seeing this sign
                if detected_sign != self.last_sign:

                    self.last_sign = detected_sign

                    self.sign_start_time = time.time()

                    self.letter_added = False

                # Same sign is being held
                else:

                    if self.sign_start_time is not None:

                        held_time = (
                            time.time()
                            - self.sign_start_time
                        )

                        # Add letter after 1 second
                        if (
                            held_time >= 1.0
                            and not self.letter_added
                        ):

                            self.word += detected_sign

                            self.letter_added = True

            # ------------------------------------------------
            # Draw hand landmarks
            # ------------------------------------------------

            for landmark in hand:

                x = int(
                    landmark.x * img.shape[1]
                )

                y = int(
                    landmark.y * img.shape[0]
                )

                cv2.circle(
                    img,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # ====================================================
        # NO HAND
        # ====================================================

        else:

            detected_sign = "No hand"
            detected_confidence = 0.0

            # Reset so next sign can be detected
            self.last_sign = ""
            self.sign_start_time = None
            self.letter_added = False

        # ====================================================
        # SAVE CURRENT RESULT
        # ====================================================

        with self.lock:

            self.current_sign = detected_sign
            self.confidence = detected_confidence

        # ====================================================
        # CAMERA OVERLAY
        # ====================================================

        # Sign
        cv2.putText(
            img,
            f"Sign: {detected_sign}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        # Confidence
        cv2.putText(
            img,
            f"Confidence: {detected_confidence:.1f}%",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        # Recognized text
        cv2.putText(
            img,
            f"Text: {self.word}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        # Instruction
        cv2.putText(
            img,
            "Hold sign for 1 sec",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Return frame
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# START CAMERA
# ============================================================

ctx = webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)

# ============================================================
# INFORMATION BELOW CAMERA
# ============================================================


# ============================================================
# HOW TO USE
# ============================================================

st.markdown("---")

st.subheader("📌 How to form a word")

st.write(
    """
    **Example: HELLO**

    1. Show H → hold for 1 second.
    2. Remove your hand.
    3. Show E → hold for 1 second.
    4. Remove your hand.
    5. Show L → hold for 1 second.
    6. Remove your hand.
    7. Show L → hold for 1 second.
    8. Remove your hand.
    9. Show O → hold for 1 second.

    The camera should display:

    **Text: HELLO**
    """
)
