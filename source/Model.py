import torchvision
import torch.nn as nn
import torch
from torchvision import transforms,models,datasets


def Model():
    model= models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    return model , loss_fn ,optimizer
