import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataset(directory="./datasets", batch_size=64, img_size=224):
    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.FER2013(root=directory, train=True,
                                 download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    test_set = datasets.FER2013(root=directory, train=False,
                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader
