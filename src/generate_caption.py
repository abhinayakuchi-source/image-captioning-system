import os
import sys
import pickle

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNet50_Weights
from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

IMAGE_PATH = "dataset/images/Images/1000268201_693b08cb0e.jpg"

MODEL_PATH = "models/image_captioning_model.pth"

VOCAB_PATH = "dataset/captions/word_to_index.pkl"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# IMPORT MODEL
# ==========================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SRC_DIR)

from model import ImageCaptioningModel


# ==========================================
# LOAD VOCABULARY
# ==========================================

with open(VOCAB_PATH, "rb") as f:
    word_to_index = pickle.load(f)

index_to_word = {
    index: word
    for word, index in word_to_index.items()
}

vocab_size = len(word_to_index)

print("Vocabulary size:", vocab_size)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = ImageCaptioningModel(
    vocab_size=vocab_size
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("✅ Trained model loaded")


# ==========================================
# LOAD RESNET50
# ==========================================

weights = ResNet50_Weights.DEFAULT

resnet = models.resnet50(
    weights=weights
)

# Remove classification layer
resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)

resnet.eval()

transform = weights.transforms()


# ==========================================
# EXTRACT IMAGE FEATURE
# ==========================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(
    image
)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)


with torch.no_grad():

    image_feature = resnet(
        image_tensor
    )

image_feature = image_feature.squeeze(0)


# ==========================================
# PREPARE LSTM
# ==========================================

with torch.no_grad():

    projected = model.image_projection(
        image_feature
    )

    # LSTM expects:
    # [num_layers, batch_size, hidden_size]

    hidden = projected.unsqueeze(0).unsqueeze(1)

    cell = torch.zeros_like(hidden)


# ==========================================
# GENERATE CAPTION
# ==========================================

word = "<start>"

caption = []

max_length = 30


for _ in range(max_length):

    word_id = word_to_index.get(
        word,
        word_to_index["<unk>"]
    )

    word_tensor = torch.tensor(
        [[word_id]],
        dtype=torch.long,
        device=DEVICE
    )

    embedding = model.embedding(
        word_tensor
    )

    with torch.no_grad():

        output, (hidden, cell) = model.lstm(
            embedding,
            (hidden, cell)
        )

        prediction = model.fc(
            output[:, -1, :]
        )

    predicted_id = prediction.argmax(
        dim=-1
    ).item()

    word = index_to_word[
        predicted_id
    ]

    if word == "<end>":
        break

    if word not in ["<start>", "<pad>"]:

        caption.append(word)


# ==========================================
# DISPLAY RESULT
# ==========================================

sentence = " ".join(caption)

print("\n================================")
print("GENERATED CAPTION")
print("================================")

print(sentence)