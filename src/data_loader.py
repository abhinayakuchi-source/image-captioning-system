import torch
from torch.utils.data import DataLoader

from dataset import ImageCaptionDataset


def collate_fn(batch):

    images = []
    captions = []

    for image, caption in batch:

        images.append(image)
        captions.append(caption)

    # Stack image features
    images = torch.stack(images)

    # Pad captions
    captions = torch.nn.utils.rnn.pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


dataset = ImageCaptionDataset(
    caption_file="dataset/captions/cleaned_captions.csv",
    feature_file="dataset/image_features.pkl"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)


if __name__ == "__main__":

    images, captions = next(iter(loader))

    print("Image feature shape:", images.shape)
    print("Caption shape:", captions.shape)