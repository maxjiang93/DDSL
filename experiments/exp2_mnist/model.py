import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Norm(nn.Module):
    def __init__(self, mean=0, std=1):
        super(Norm, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

class LeNet5(nn.Module):
    def __init__(self, signal_sizes=(28,28), mean=0, std=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.norm = Norm(mean, std)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 250)
        self.fc2 = nn.Linear(250, 10)
        self.signal_sizes = signal_sizes

    def forward(self, x):
        x = x.view(-1, 1, self.signal_sizes[0], self.signal_sizes[1])
        x = self.norm(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x