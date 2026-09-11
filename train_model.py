import os
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Signs we are recognizing
SIGNS = ["A", "B", "C", "D", "E"]

X = []
y = []


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")
print()

for label, sign in enumerate(SIGNS):

    folder = f"dataset/{sign}"

    for filename in os.listdir(folder):

        if filename.endswith(".csv"):

            filepath = os.path.join(folder, filename)

            with open(filepath, "r") as file:
                reader = csv.reader(file)
                landmarks = next(reader)

            landmarks = [float(value) for value in landmarks]

            X.append(landmarks)
            y.append(label)

    print(f"Loaded {sign} samples")


X = np.array(X)
y = np.array(y)

print()
print("Dataset shape:", X.shape)


# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# SCALE DATA
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# CONVERT LABELS
# =========================

y_train = to_categorical(y_train, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)


# =========================
# CREATE NEURAL NETWORK
# =========================

model = Sequential([

    Dense(128, activation="relu", input_shape=(63,)),

    Dropout(0.2),

    Dense(64, activation="relu"),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(5, activation="softmax")
])


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# TRAIN MODEL
# =========================

print()
print("Starting training...")
print()

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)


# =========================
# TEST MODEL
# =========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("==============================")
print("TRAINING COMPLETED!")
print("==============================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

model.save("models/sign_language_model.keras")

np.save("models/scaler_mean.npy", scaler.mean_)
np.save("models/scaler_scale.npy", scaler.scale_)

print()
print("Model saved successfully!")
print("Location: models/sign_language_model.keras")