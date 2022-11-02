import torch
import torch.nn as nn
from dataloader import get_dataset
from train import Trainer
import FERVT
import os
from torch.optim import Adam, AdamW
from VIT import MyViT
from VITT import VIT
import math
from vit_pytorch.deepvit import DeepViT

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
    FER_VT = DeepViT(image_size = 48, patch_size = 8, num_classes = num_classes, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1).to(device)
    # FER_VT = VIT(img_size= (48,48),patch_size= (8,8), emb_dim = 1280, mlp_dim=5120, num_heads=16, num_layers=32, n_classes=num_classes, dropout_rate=0.1, at_d_r=0.0).to(device)
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
