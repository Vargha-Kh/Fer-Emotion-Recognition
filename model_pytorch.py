import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Regnet(nn.Module):
    def __int__(self, pretrained_model=True, fine_tune=True, num_classes=7):
        super().__init__()
        self.pretrained = pretrained_model
        self.fine_tune = fine_tune
        self.num_classes = num_classes
        self.num_features = None

    def get_model(self):
        model_pt = models.regnet_y_400mf(pretrained=self.pretrained_model)
        set_parameter_requires_grad(model_pt, True)
        # self.num_features = model_ft.AuxLogits.fc.in_features
        self.num_features = model_pt.fc.in_features
        model_pt.fc = nn.Sequential(
            nn.Linear(in_features=self.num_features, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=self.num_classes)
        )
        self.pretrained = model_pt
        return model_pt
    #
    # def forward(self, x):
    #     x = self.pretrained(x)
    #     x = torch.flatten(x, 1)
    #     logit = self.newlayers(x)
    #     prob = F.softmax(logit, dim=1)
    #     return logit, prob
