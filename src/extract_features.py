import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import pickle


IMAGE_DIR = "dataset/images/Images"
OUTPUT_FILE = "dataset/image_features.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# Load ResNet50
print("Loading ResNet50...")

weights = ResNet50_Weights.DEFAULT

resnet = models.resnet50(weights=weights)

# Remove classification layer
resnet.fc = torch.nn.Identity()

resnet = resnet.to(DEVICE)
resnet.eval()

transform = weights.transforms()

print("✅ ResNet50 loaded")

# Get images
image_files = [
    file for file in os.listdir(IMAGE_DIR)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Images found:", len(image_files))

features = {}

print("\nExtracting features...")

with torch.no_grad():

    for image_name in tqdm(image_files):

        image_path = os.path.join(IMAGE_DIR, image_name)

        try:
            image = Image.open(image_path).convert("RGB")

            image = transform(image)
            image = image.unsqueeze(0).to(DEVICE)

            feature = resnet(image)

            feature = feature.squeeze().cpu()

            features[image_name] = feature

        except Exception as e:
            print(f"\n❌ Error: {image_name}")
            print(e)


# Save
print("\nSaving features...")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(features, f)

print("✅ Feature extraction completed!")
print("Features saved:", len(features))
print("File:", OUTPUT_FILE)