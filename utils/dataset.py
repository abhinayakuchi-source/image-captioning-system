import os
from PIL import Image
from torch.utils.data import Dataset


class FlickrDataset(Dataset):

    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.images = []
        self.captions = []

        with open(caption_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                image_name, caption = line.split(",", 1)

                self.images.append(image_name)
                self.captions.append(caption)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]
        caption = self.captions[index]

        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption