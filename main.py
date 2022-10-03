import torch
import torch.nn as nn
from dataloader import get_dataset
from model_pytorch import Regnet
from train import Trainer

if __name__ == "__main__":
    num_classes = 7
    augmentation = True
    batch_size = 32
    num_epochs = 300
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Prepare
    train_loader, val_loader = get_dataset()

    # Model loading
    regnet = Regnet()
    model = regnet.get_model()
    model.to(device)

    # Hyper-parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(model, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, device, 50)
