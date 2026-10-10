import torch
import torch.nn as nn
from torchvision import models


class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Загружаем предобученную ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        original_conv1 = self.resnet.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))

        self.resnet.conv1 = new_conv1

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)