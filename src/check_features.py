import pandas as pd
import pickle
import os

CAPTION_FILE = "dataset/captions/cleaned_captions.csv"
FEATURE_FILE = "dataset/image_features.pkl"

# Load captions
df = pd.read_csv(CAPTION_FILE)

# Load extracted features
with open(FEATURE_FILE, "rb") as f:
    features = pickle.load(f)

print("Number of captions:", len(df))
print("Number of extracted features:", len(features))

# Find images mentioned in captions
caption_images = set(df["image"])

# Find missing features
missing = caption_images - set(features.keys())

print("\nImages referenced by captions:", len(caption_images))
print("Images with features:", len(features))
print("Missing features:", len(missing))

print("\nFirst 20 missing images:")

for image in list(missing)[:20]:
    print(image)