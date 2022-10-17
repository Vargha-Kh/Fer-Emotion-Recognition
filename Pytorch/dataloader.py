import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataset(directory="./fer2013", batch_size=128, img_size=224):
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(0.2),
         transforms.RandomRotate(0.1),
        transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(directory + '/train', transform=transform)
    val_data = datasets.ImageFolder(directory + '/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader, 

