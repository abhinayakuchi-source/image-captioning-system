import os
import sys
import pickle
import random

import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNet50_Weights
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/best_model.pth"
CAPTION_FILE = "dataset/captions/cleaned_captions.csv"
VOCAB_PATH = "dataset/captions/word_to_index.pkl"
IMAGE_DIR = "dataset/images"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_IMAGES = 100


# ==========================================
# IMPORT MODEL
# ==========================================

SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src"
)

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
# LOAD MODEL
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

print("Loading ResNet50...")

weights = ResNet50_Weights.DEFAULT

resnet = models.resnet50(
    weights=weights
)

resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)
resnet.eval()

transform = weights.transforms()

print("✅ ResNet50 loaded")


# ==========================================
# LOAD CAPTIONS
# ==========================================

import pandas as pd

data = pd.read_csv(CAPTION_FILE)

print("Total captions:", len(data))


# ==========================================
# GROUP CAPTIONS BY IMAGE
# ==========================================

grouped = data.groupby("image")["caption"].apply(list).to_dict()

image_names = list(grouped.keys())

# Reproducible selection
random.seed(42)
random.shuffle(image_names)

image_names = image_names[:NUM_IMAGES]

print("Images selected for evaluation:", len(image_names))


# ==========================================
# CAPTION GENERATION FUNCTION
# ==========================================

def generate_caption(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        image_feature = resnet(
            image_tensor
        )

        image_feature = image_feature.squeeze(0)

        projected = model.image_projection(
            image_feature
        )

        hidden = projected.unsqueeze(0).unsqueeze(1)

        cell = torch.zeros_like(hidden)

    word = "<start>"

    generated = []

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

        word = index_to_word.get(
            predicted_id,
            "<unk>"
        )

        if word == "<end>":
            break

        if word not in [
            "<start>",
            "<pad>"
        ]:

            generated.append(word)

    return generated


# ==========================================
# BLEU EVALUATION
# ==========================================

smooth = SmoothingFunction().method1

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

successful = 0


print()
print("======================================")
print("STARTING BLEU EVALUATION")
print("======================================")


for count, image_name in enumerate(
    image_names,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    if not os.path.exists(image_path):
        continue

    try:

        generated = generate_caption(
            image_path
        )

        references = [
            caption.lower().split()
            for caption in grouped[image_name]
        ]

        if not generated:
            continue

        bleu1 = sentence_bleu(
            references,
            generated,
            weights=(1, 0, 0, 0),
            smoothing_function=smooth
        )

        bleu2 = sentence_bleu(
            references,
            generated,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smooth
        )

        bleu3 = sentence_bleu(
            references,
            generated,
            weights=(1/3, 1/3, 1/3, 0),
            smoothing_function=smooth
        )

        bleu4 = sentence_bleu(
            references,
            generated,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth
        )

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

        successful += 1

        if count % 10 == 0:

            print(
                f"Processed {count}/{len(image_names)}"
            )

    except Exception as e:

        print(
            f"Error processing {image_name}: {e}"
        )


# ==========================================
# RESULTS
# ==========================================

print()
print("======================================")
print("BLEU EVALUATION RESULTS")
print("======================================")

if successful > 0:

    print(
        f"Images evaluated: {successful}"
    )

    print(
        f"BLEU-1: {sum(bleu1_scores) / len(bleu1_scores):.4f}"
    )

    print(
        f"BLEU-2: {sum(bleu2_scores) / len(bleu2_scores):.4f}"
    )

    print(
        f"BLEU-3: {sum(bleu3_scores) / len(bleu3_scores):.4f}"
    )

    print(
        f"BLEU-4: {sum(bleu4_scores) / len(bleu4_scores):.4f}"
    )

else:

    print("No images were successfully evaluated.")
