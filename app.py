import os
import sys
import pickle

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNet50_Weights
from PIL import Image


# ==========================================
# SETTINGS
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/image_captioning_model.pth"

VOCAB_PATH = "dataset/captions/word_to_index.pkl"


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
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="centered"
)


st.title("🖼️ AI Image Captioning")
st.write(
    "Upload an image and let the deep learning model generate a caption."
)


# ==========================================
# LOAD VOCABULARY
# ==========================================

@st.cache_resource
def load_vocabulary():

    with open(
        VOCAB_PATH,
        "rb"
    ) as f:

        word_to_index = pickle.load(f)

    index_to_word = {
        index: word
        for word, index in word_to_index.items()
    }

    return word_to_index, index_to_word


# ==========================================
# LOAD LSTM MODEL
# ==========================================

@st.cache_resource
def load_caption_model():

    word_to_index, _ = load_vocabulary()

    vocab_size = len(word_to_index)

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

    return model


# ==========================================
# LOAD RESNET50
# ==========================================

@st.cache_resource
def load_resnet():

    weights = ResNet50_Weights.DEFAULT

    resnet = models.resnet50(
        weights=weights
    )

    # Remove classification layer
    resnet.fc = nn.Identity()

    resnet = resnet.to(DEVICE)

    resnet.eval()

    transform = weights.transforms()

    return resnet, transform


# ==========================================
# GENERATE CAPTION
# ==========================================

def generate_caption(
    image,
    model,
    resnet,
    transform,
    word_to_index,
    index_to_word
):

    # Preprocess image
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)


    # Extract image features
    with torch.no_grad():

        image_feature = resnet(
            image_tensor
        )


    # Project image feature
    with torch.no_grad():

        projected = model.image_projection(
            image_feature
        )

        hidden = projected.unsqueeze(0)

        cell = torch.zeros_like(hidden)


    # Start caption
    current_word = "<start>"

    caption_words = []

    max_length = 30


    # Generate words
    for _ in range(max_length):

        word_id = word_to_index.get(
            current_word,
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


        current_word = index_to_word.get(
            predicted_id,
            "<unk>"
        )


        if current_word == "<end>":
            break


        if current_word not in [
            "<start>",
            "<pad>"
        ]:

            caption_words.append(
                current_word
            )


    return " ".join(caption_words)


# ==========================================
# LOAD MODELS
# ==========================================

try:

    word_to_index, index_to_word = load_vocabulary()

    caption_model = load_caption_model()

    resnet, transform = load_resnet()

except Exception as e:

    st.error(
        f"Could not load the model: {e}"
    )

    st.stop()


# ==========================================
# IMAGE UPLOADER
# ==========================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    if st.button("✨ Generate Caption"):

        with st.spinner(
            "Generating caption..."
        ):

            caption = generate_caption(
                image,
                caption_model,
                resnet,
                transform,
                word_to_index,
                index_to_word
            )


        st.success(
            "Caption generated!"
        )

        st.subheader("Generated Caption")

        st.write(
            f"**{caption}**"
        )