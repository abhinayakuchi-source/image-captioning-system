import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import pickle
from dataset import ImageCaptionDataset
from data_loader import collate_fn
from model import ImageCaptioningModel

from torch.utils.data import DataLoader


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -------------------------
# Load vocabulary
# -------------------------

with open(
    "dataset/captions/word_to_index.pkl",
    "rb"
) as f:
    word_to_index = pickle.load(f)

vocab_size = len(word_to_index)

print("Vocabulary size:", vocab_size)


# -------------------------
# Dataset
# -------------------------

dataset = ImageCaptionDataset(
    caption_file="dataset/captions/cleaned_captions.csv",
    feature_file="dataset/image_features.pkl"
)

print("Dataset size:", len(dataset))


# -------------------------
# DataLoader
# -------------------------

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)


# -------------------------
# Get one batch
# -------------------------

images, captions = next(iter(loader))

print("Image features:", images.shape)
print("Captions:", captions.shape)


# -------------------------
# Create model
# -------------------------

model = ImageCaptioningModel(
    vocab_size=vocab_size
)

model = model.to(device)

images = images.to(device)
captions = captions.to(device)


# -------------------------
# Forward pass
# -------------------------

with torch.no_grad():

    outputs = model(
        images,
        captions
    )


print("Model output:", outputs.shape)

print("\n✅ MODEL TEST PASSED!")