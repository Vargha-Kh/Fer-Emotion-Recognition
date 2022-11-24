import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imgaug.augmenters as iaa
import numpy as np


def augmentation(data_dir):
    augmented = list()
    s = iaa.Sequential([
        iaa.PiecewiseAffine(scale=0.05),
        iaa.Rotate((-30, 30)),
        iaa.TranslateX(px=(-30, 30)),
        iaa.TranslateY(px=(-30, 30)),
        iaa.Grayscale(0.6),
        iaa.Fliplr(1.0),
        iaa.Crop(percent=0.02),
        iaa.ChangeColorTemperature((6000, 11000))
    ])
    train_transforms = transforms.Compose([transforms.Resize((48, 48))])
    images = datasets.ImageFolder(data_dir, transform=train_transforms)
    classes = datasets.ImageFolder(data_dir).class_to_idx
    for img in images:
        img_ = list(img)
        for _ in classes:
            tf = transforms.Compose([s.augment_image, transforms.ToTensor()])
            tf(np.array(img_[0]))
    augmented.append(tf)
    return augmented


def get_dataset(directory="./fer2013", batch_size=128, img_size=48):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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

    train_data = datasets.ImageFolder(directory + '/train', transform=transform_train)
    val_data = datasets.ImageFolder(directory + '/val', transform=transform_val)

    augmented_data = augmentation(directory + '/train')
    augmented_data.append(train_data)
    train_data = torch.utils.data.ConcatDataset(augmented_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader,
