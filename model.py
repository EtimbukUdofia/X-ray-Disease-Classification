# model.py
import torch.nn as nn
from torchvision import models


def get_model(num_classes=6):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
