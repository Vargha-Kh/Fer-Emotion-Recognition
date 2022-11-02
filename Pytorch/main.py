import torch
import torch.nn as nn
from dataloader import get_dataset
from model_pytorch import Regnet
from train import Trainer
import FERVT
from adan_pytorch import Adan
from adabelief_pytorch import AdaBelief
import os
from torch.optim import Adam, AdamW
from VIT import MyViT
from VITT import VIT
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()

if __name__ == "__main__":
    num_classes = 7
    augmentation = True
    batch_size = 128
    num_epochs = 300

    # Data Prepare
    train_loader, val_loader = get_dataset()

    # Model loading
    FER_VT = VIT().to(device)
    #FER_VT = MyViT((1, 48, 48), n_patches=8, n_blocks=4, hidden_d=128, n_heads=8, out_d=num_classes).to(device)
    # Hyper-parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(FER_VT.parameters(), lr=0.001, weight_decay=0.00001)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                   lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (
                                                           1 - 0.2) + 0.2)
#    FER_VT.load_state_dict(torch.load('./model/best.pth'))
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(FER_VT, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, scheduler, device, num_epochs)
