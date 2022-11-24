import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_augmentation(aug_name, img_h, img_w, mean, std, **kwargs):
    augmentations = dict(
        normal=get_normal_aug,
    )
    return augmentations[aug_name](img_h=img_h, img_w=img_w, mean=mean, std=std, **kwargs)


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

    train_data = datasets.ImageFolder(directory + '/train', transform=train_transform)
    val_data = datasets.ImageFolder(directory + '/val', transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader,
