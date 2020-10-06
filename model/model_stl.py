# model for STL 10

import torch.nn as nn
import torch.nn.functional as F


class NetworkPhi(nn.Module):
    def __init__(self):
        super(NetworkPhi, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.sigmoid = nn.Sigmoid()
        self.LogSoftMax = nn.LogSoftmax(dim=1)
        self.m = nn.Dropout2d(0.2)
        self.n = nn.Dropout(0.2)
        self.b1 = nn.BatchNorm2d(6)
        self.b2 = nn.BatchNorm2d(16)
        self.b3 = nn.BatchNorm1d(120)
        self.b4 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.LogSoftMax(x)
