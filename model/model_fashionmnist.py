# model for FashionMNIST

import torch.nn as nn


class NetworkPhi(nn.Module):
    def __init__(self):
        super(NetworkPhi, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)
        self.fc2 = nn.Linear(84, 2)
        self.sigmoid = nn.Sigmoid()
        self.LogSoftMax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.bn_fc1(self.fc1(out)))
        out = self.fc2(out)
        return self.LogSoftMax(out)
