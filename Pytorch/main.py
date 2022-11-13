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
from vit_pytorch.deepvit import DeepViT
from networks import resnet18_at
from Lookahead import Lookahead
from torch_lr_finder import LRFinder


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
    FER_VT = DeepViT(image_size = 48, patch_size = 8, num_classes = num_classes, dim = 1280, depth = 4, heads = 16, mlp_dim = 5120, dropout = 0.1, emb_dropout = 0.1).to(device)
    #FER_VT = VIT(img_size= (48,48),patch_size= (8,8), emb_dim = 2048, mlp_dim=5120, num_heads=16, num_layers=32, n_classes=num_classes, dropout_rate=0.25, at_d_r=0.1).to(device)
    #FER_VT = MyViT((1, 48, 48), n_patches=8, n_blocks=4, hidden_d=1280, n_heads=16, out_d=num_classes).to(device)
    #FER_VT = resnet18_at(num_classes=7, at_type=0).to(device)

    # Hyper-parameters
    wd = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(FER_VT.parameters(), lr=0.001, weight_decay=wd)
    #optimizer = SGDW(FER_VT.parameters(), lr=0.001, momentum=9, weight_decay=wd)
    #exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, verbose=True)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
#                                                   lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (
 #                                                          1 - 0.2) + 0.2)

    # lr_finder = LRFinder(FER_VT, optimizer, criterion, device="cuda")
    # lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=0.1, num_iter=100, step_mode="linear")
    # print("LR: ", lr_finder)

    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.001, 0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, T_mult=1, eta_min=0.1, last_epoch=- 1, verbose=True)
    #FER_VT.load_state_dict(torch.load('./model/best.pth'))
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
    #optimizer = Lookahead( optimizer, alpha= 0.6 , k = 10) 
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    trainer = Trainer(logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model")
    model, optimizer, train_loss, valid_loss = trainer.training(FER_VT, train_loader, val_loader, criterion, optimizer,
                                                                reduce_on_plateau, lr_scheduler, device, num_epochs)
