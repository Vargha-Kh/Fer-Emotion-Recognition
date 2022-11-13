import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataset(directory="./fer2013", batch_size=128, img_size=48):
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    transform_train = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.Grayscale(),
         transforms.RandomRotation(0.3),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    transform_val = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])

    train_data = datasets.ImageFolder(directory + '/train', transform=transform_train)
    val_data = datasets.ImageFolder(directory + '/val', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader,
