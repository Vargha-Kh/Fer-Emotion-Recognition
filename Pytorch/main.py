import torch
import warnings
import torch.nn as nn
from dataloader import get_dataset
from train import Trainer
import FERVT
import os
import numpy as np
from torch.optim import AdamW

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()

if __name__ == "__main__":
    num_classes = 7
    augmentation = True
    batch_size = 256
    num_epochs = 300
    img_size = 224

    # Data Prepare
    train_loader, val_loader = get_dataset(directory="./fer2013", batch_size=batch_size, img_size=img_size)

    # Model loading
    FER_VT = FERVT.FERVT(device)

    # Hyper-parameters
    wd = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(FER_VT.parameters(), lr=0.001, weight_decay=wd)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, T_mult=1, eta_min=0.0001,
    #                                                                     last_epoch=- 1, verbose=True)
    FER_VT.load_state_dict(torch.load('./model/best.pth'))
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(FER_VT, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, lr_scheduler, device, num_epochs)
