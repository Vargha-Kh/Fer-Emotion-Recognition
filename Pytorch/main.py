import torch
import torch.nn as nn
from dataloader import get_dataset
from model_pytorch import Regnet
from train import Trainer
import FERVT
import math
from adan_pytorch import Adan
from adabelief_pytorch import AdaBelief
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# torch.cuda.empty_cache()

if __name__ == "__main__":
    num_classes = 7
    augmentation = True
    batch_size = 256
    num_epochs = 300

    # Data Prepare
    train_loader, val_loader = get_dataset()

    # Model loading
    FER_VT = FERVT.FERVT(device)

    # Hyper-parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adan(FER_VT.parameters(), lr=0.001, betas = (0.02, 0.08, 0.01), weight_decay=0.02)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (
                                                          1 - 0.2) + 0.2)
    # FER_VT.load_state_dict(torch.load('./model/resnet_best.pth'))
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(FER_VT, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, device, num_epochs)
