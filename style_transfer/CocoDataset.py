import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CocoDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image)

        return tensor_image
