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
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_address = self.images[idx]
        image = cv2.imread(img_address)[..., ::-1]
        image = self.transform(image=image)['image']
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

    train_data = ImageDataset(os.path.join(directory, '/train'), transform=train_transform)
    val_data = ImageDataset(os.path.join(directory + '/val'), transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader,
