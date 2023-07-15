import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader



class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2), # 10x3x32x32 -> 10x6x32x32
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 10x6x32x32 -> 10x6x16x16
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), # 10x6x16x16 -> 10x16x12x12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 10x16x12x12 -> 10x16x6x6
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(inplace=True), # inplace=True means that it will modify the input directly
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2), # 10x3x32x32 -> 10x64x17x17
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 10x64x17x17 -> 10x64x8x8
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2), #  10x64x8x8 -> 10x192x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 10x192x8x8 -> 10x192x3x3

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # 10x192x3x3 -> 10x384x3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), # 10x384x3x3 -> 10x256x3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 10x256x3x3 -> 10x256x3x3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 10x256x3x3 -> 10x256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LinearNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LinearNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x