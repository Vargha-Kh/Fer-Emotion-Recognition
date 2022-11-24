import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from deep_utils import crawl_directory_dataset
import torch.nn.functional as F
from torchvision.io import read_image


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, n_classes=7):
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_address = self.images[idx]
        # img = read_image(img_address)
        # img = self.transform(image=img)['image']
        img = cv2.imread(img_address, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # image_np = np.array(img)
        # Apply transformations
        augmented = self.transform(image=img)
        image = augmented['image']
        # Convert numpy array to PIL Image
        img = Image.fromarray(augmented['image'])

        label = torch.tensor(self.labels[idx])
        label = F.one_hot(label, num_classes=self.n_classes)
        sample = (image, label)

        return sample


def get_dataset(directory="./fer2013", batch_size=128, img_size=48):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            # A.ToGray(always_apply=True, p=1),
            # A.ToRGB(always_apply=True, p=1),
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

    EMOTION_ID2NAME = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    EMOTION_NAME2ID = {v: k for k, v in EMOTION_ID2NAME.items()}

    train_address, train_labels = crawl_directory_dataset(directory + '/train',
                                                          label_map_dict=EMOTION_NAME2ID)
    val_address, val_labels = crawl_directory_dataset(directory + '/val', label_map_dict=EMOTION_NAME2ID)
    train_dataset = CustomDataset(train_address, train_labels, transform=train_transform)
    val_dataset = CustomDataset(val_address, val_labels, transform=train_transform)

    # train_dataset = datasets.ImageFolder(dataset_dir + '/train', transform=transform_train)
    # val_dataset = datasets.ImageFolder(dataset_dir + '/val', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    return train_loader, val_loader,
