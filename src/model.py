import torch
import torch.nn as nn


class ImageCaptioningModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512,
        feature_size=2048
    ):
        super().__init__()

        self.image_projection = nn.Linear(
            feature_size,
            hidden_size
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, images, captions):

        image_features = self.image_projection(images)

        hidden = image_features.unsqueeze(0)

        cell = torch.zeros_like(hidden)

        captions_input = captions[:, :-1]

        embeddings = self.embedding(captions_input)

        output, _ = self.lstm(
            embeddings,
            (hidden, cell)
        )

        output = self.fc(output)

        return output