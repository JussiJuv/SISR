import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class PairedImageDataset(Dataset):
    def __init__(self, hr_path, lr_path, transform=None, image_size=128):
        self.hr_path = hr_path
        if "X8" in lr_path:
            self.scale = 8
        elif "X4" in lr_path:
            self.scale = 4
        else:
            self.scale = 4
        self.lr_path = lr_path

        self.filenames = sorted([f for f in os.listdir(hr_path) if f.endswith('.png')])
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_name = self.filenames[idx]
        lr_name = hr_name.replace(".png", f"x{self.scale}.png")
        
        hr_img = Image.open(os.path.join(self.hr_path, hr_name)).convert("RGB")
        
        # Find the LR image
        lr_full_path = os.path.join(self.lr_path, lr_name)
        if not os.path.exists(lr_full_path):
            lr_full_path = os.path.join(self.lr_path, hr_name)
        lr_img = Image.open(lr_full_path).convert("RGB")

        # Check if we are in 'train' or 'valid' based on the folder path
        is_train = "train" in self.hr_path.lower()

        if is_train:
            th, tw = self.image_size, self.image_size
            lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)
            w, h = hr_img.size
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            hr_img = F.crop(hr_img, i, j, th, tw)
            lr_img = F.crop(lr_img, i, j, th, tw)
        else:
            if self.image_size is not None:
                th, tw = self.image_size, self.image_size
                hr_img = F.resize(hr_img, (th, tw), interpolation=Image.LANCZOS)
                lr_img = F.resize(lr_img, (th, tw), interpolation=Image.LANCZOS)
            else:
                if lr_img.size != hr_img.size:
                    lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

        # Apply standard transforms
        hr_tensor = self.transform(hr_img) if self.transform else F.to_tensor(hr_img)
        lr_tensor = self.transform(lr_img) if self.transform else F.to_tensor(lr_img)
        
        # Final Normalization to [-1, 1]
        hr_tensor = F.normalize(hr_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        lr_tensor = F.normalize(lr_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        return hr_tensor, lr_tensor, hr_name