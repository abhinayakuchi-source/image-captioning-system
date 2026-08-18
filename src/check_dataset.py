import os
import pandas as pd

IMAGE_DIR = "dataset/Images"
CAPTION_FILE = "dataset/captions/captions.txt"

print("Checking dataset...\n")

# Check Images folder
if not os.path.exists(IMAGE_DIR):
    print("❌ Images folder not found!")
    exit()

images = [
    file for file in os.listdir(IMAGE_DIR)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("✅ Images folder found")
print("Number of images:", len(images))

# Check captions file
if not os.path.exists(CAPTION_FILE):
    print("❌ Captions file not found!")
    exit()

print("✅ Captions file found")

# Read captions
df = pd.read_csv(
    CAPTION_FILE,
    sep=",",
    header=None,
    names=["image", "caption"]
)

print("Number of captions:", len(df))

print("\nFirst 5 captions:")
print(df.head())

# Check whether caption images exist
image_names = set(images)

df["image_exists"] = df["image"].isin(image_names)

matched = df["image_exists"].sum()

print("\nCaptions with matching images:", matched)
print("Captions without images:", len(df) - matched)