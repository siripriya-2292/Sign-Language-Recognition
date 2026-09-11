import cv2
import mediapipe as mp

# Create MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# Start the hand landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera.")
            break

        # Convert OpenCV image (BGR) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = landmarker.detect(mp_image)

        # Draw hand landmarks
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

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

        # Display camera
        cv2.imshow(
            "Sign Language - Hand Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()