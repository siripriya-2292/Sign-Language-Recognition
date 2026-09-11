import cv2
import mediapipe as mp
import os
import csv
import time

SIGNS = ["A", "B", "C", "D", "E"]
SAMPLES_PER_SIGN = 100

# Create folders
for sign in SIGNS:
    os.makedirs(f"dataset/{sign}", exist_ok=True)

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

    for sign in SIGNS:

        print()
        print("==============================")
        print("Get ready for SIGN:", sign)
        print("==============================")

        input(f"Press ENTER when you are ready to show sign {sign}...")

        sample_count = 0

        while sample_count < SAMPLES_PER_SIGN:

            success, frame = cap.read()

            if not success:
                print("Could not read camera.")
                break

            # Mirror the camera
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

                # Save data
                filename = f"dataset/{sign}/{sign}_{sample_count}.csv"

                with open(filename, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(landmarks)

                sample_count += 1

                print(f"{sign}: {sample_count}/{SAMPLES_PER_SIGN}")

                # Small delay so samples are not identical
                time.sleep(0.1)

            # Display information
            cv2.putText(
                frame,
                f"SHOW SIGN: {sign}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {sample_count}/{SAMPLES_PER_SIGN}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Sign Language Data Collection", frame)

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Finished collecting {sign}!")

    cap.release()
    cv2.destroyAllWindows()

print()
print("====================================")
print("DATA COLLECTION COMPLETED!")
print("====================================")
print("A = 100 samples")
print("B = 100 samples")
print("C = 100 samples")
print("D = 100 samples")
print("E = 100 samples")
print("Total = 500 samples")