import torch
import torch.nn as nn
from dataloader import get_dataset
from train import Trainer
import FERVT
import os
from SGDW import SGDW
from torch.optim import Adam, AdamW, SGD
from VIT import MyViT
from VITT import VIT
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.cuda.empty_cache()

if __name__ == "__main__":
    num_classes = 7
    augmentation = True
    batch_size = 512
    num_epochs = 300
    img_size = 48
    # Data Prepare
    train_loader, val_loader = get_dataset(directory="./fer2013", batch_size=batch_size, img_size=img_size)

    # Model loading
    FER_VT = VIT(img_size=(3, 48, 48), patch_size=(8, 8), emb_dim=1024, mlp_dim=2048, num_heads=8, num_layers=24,
                 n_classes=num_classes, dropout_rate=0.25, at_d_r=0.25).to(device)
    # FER_VT = MyViT((1, 48, 48), n_patches=8, n_blocks=4, hidden_d=1280, n_heads=16, out_d=num_classes).to(device)

    # Hyper-parameters
    wd = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(FER_VT.parameters(), lr=0.001, weight_decay=wd)

    # exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, T_mult=1, eta_min=0.0001,
    #                                                                     last_epoch=- 1, verbose=True)
    # FER_VT.load_state_dict(torch.load('./model/best.pth'))
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, verbose=True)
    # optimizer = Lookahead( optimizer, alpha= 0.6 , k = 10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(FER_VT, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, lr_scheduler, device, num_epochs)
