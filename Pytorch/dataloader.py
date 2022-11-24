import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset
import cv2
import os


class ImageDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image


def get_dataset(directory="./fer2013", batch_size=128, img_size=48):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ToGray(always_apply=True, p=1),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2()
        ])

    transform_train = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.Grayscale(num_output_channels=3),
         transforms.RandomRotation(0.3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    transform_val = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.Grayscale(num_output_channels=3),
         transforms.RandomRotation(0.3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    train_path = "./fer2013/train"
    train_data = ImageDataset(train_path, transform=train_transform)
    val_data = ImageDataset(os.path.join(directory, '/val'), transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader,
