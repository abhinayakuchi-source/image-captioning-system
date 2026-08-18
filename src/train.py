import os
import sys
import pickle

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

# ==========================================
# IMPORT SRC FILES
# ==========================================

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from dataset import ImageCaptionDataset
from data_loader import collate_fn
from model import ImageCaptioningModel


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("IMAGE CAPTIONING TRAINING")
print("======================================")

print("Using device:", DEVICE)


# ==========================================
# LOAD VOCABULARY
# ==========================================

with open(
    "dataset/captions/word_to_index.pkl",
    "rb"
) as f:

    word_to_index = pickle.load(f)

vocab_size = len(word_to_index)

print("Vocabulary size:", vocab_size)


# ==========================================
# LOAD DATASET
# ==========================================

dataset = ImageCaptionDataset(
    caption_file="dataset/captions/cleaned_captions.csv",
    feature_file="dataset/image_features.pkl"
)

print("Total samples:", len(dataset))


# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ==========================================
# CREATE MODEL
# ==========================================

model = ImageCaptioningModel(
    vocab_size=vocab_size
)

model = model.to(DEVICE)

print("✅ Model created")


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# SAVE DIRECTORY
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


BEST_MODEL_PATH = (
    "models/best_model.pth"
)


# ==========================================
# HISTORY
# ==========================================

train_losses = []

val_losses = []

best_val_loss = float("inf")


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    print()
    print(
        f"========== Epoch {epoch + 1}/{EPOCHS} =========="
    )

    # --------------------------------------
    # TRAIN
    # --------------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_number, (images, captions) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)

        captions = captions.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            images,
            captions
        )

        # Target words
        targets = captions[:, 1:]

        # Reshape
        outputs = outputs.reshape(
            -1,
            vocab_size
        )

        targets = targets.reshape(-1)

        # Loss
        loss = criterion(
            outputs,
            targets
        )

        # Backpropagation
        loss.backward()

        # Update
        optimizer.step()

        running_train_loss += loss.item()

        # Progress
        if (batch_number + 1) % 100 == 0:

            print(
                f"Batch "
                f"{batch_number + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )


    # Average training loss

    train_loss = (
        running_train_loss
        / len(train_loader)
    )

    train_losses.append(
        train_loss
    )


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, captions in val_loader:

            images = images.to(DEVICE)

            captions = captions.to(DEVICE)

            outputs = model(
                images,
                captions
            )

            targets = captions[:, 1:]

            outputs = outputs.reshape(
                -1,
                vocab_size
            )

            targets = targets.reshape(-1)

            loss = criterion(
                outputs,
                targets
            )

            running_val_loss += loss.item()


    val_loss = (
        running_val_loss
        / len(val_loader)
    )

    val_losses.append(
        val_loss
    )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print()
    print(
        f"Epoch {epoch + 1} Results:"
    )

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "✅ New best model saved!"
        )


# ==========================================
# SAVE FINAL MODEL
# ==========================================

FINAL_MODEL_PATH = (
    "models/final_model.pth"
)

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH
)

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")

print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Final model:",
    FINAL_MODEL_PATH
)


# ==========================================
# PLOT LOSS
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    train_losses,
    label="Training Loss",
    marker="o"
)

plt.plot(
    val_losses,
    label="Validation Loss",
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/loss_curve.png"
)


print(
    "✅ Loss graph saved:"
    " models/loss_curve.png"
)