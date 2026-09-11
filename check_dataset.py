import os

SIGNS = ["A", "B", "C", "D", "E"]

print("Dataset Check")
print("====================")

total = 0

for sign in SIGNS:
    folder = f"dataset/{sign}"

    files = [
        file for file in os.listdir(folder)
        if file.endswith(".csv")
    ]

    count = len(files)
    total += count

    print(f"{sign}: {count} samples")

print("====================")
print(f"Total samples: {total}")