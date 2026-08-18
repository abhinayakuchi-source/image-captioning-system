import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):

    def __init__(self, caption_file, feature_file):

        self.data = pd.read_csv(caption_file)

        with open(feature_file, "rb") as f:
            self.features = pickle.load(f)

        with open("dataset/captions/word_to_index.pkl", "rb") as f:
            self.word_to_index = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["image"]
        caption = row["caption"]

        # Get image feature
        image_feature = self.features[image_name]

        # Convert caption words to numbers
        tokens = caption.split()

        caption_ids = [
            self.word_to_index.get(word, self.word_to_index["<unk>"])
            for word in tokens
        ]

        caption_ids = torch.tensor(
            caption_ids,
            dtype=torch.long
        )

        return image_feature, caption_ids