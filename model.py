import torch
import torch.nn as nn


class SwishModel(nn.Module):
    def __init__(self, image_size):
        super(SwishModel, self).__init__()

        self.image_size = image_size
        size = self._get_feature_size()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout2 = nn.Dropout2d(p=0.2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(size * size * 256, 128)
        self.dropout3 = nn.Dropout(p=0.20)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(p=0.20)
        self.fc3 = nn.Linear(64, 32)
        self.swish = nn.SiLU()

    def _get_feature_size(self):
        size = self.image_size
        for _ in range(4):
            size = (size - 3) + 1
            size = (size - 2) // 2 + 1
        return size

    def forward(self, x):
        x = self.pool1(self.swish(self.bn1(self.conv1(x))))
        x = self.pool2(self.swish(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.pool3(self.swish(self.bn3(self.conv3(x))))
        x = self.pool4(self.swish(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        x = self.flatten(x)

        x = self.swish(self.dropout3(self.fc1(x)))
        x = self.swish(self.dropout4(self.fc2(x)))
        x = self.swish(self.fc3(x))
        return x

