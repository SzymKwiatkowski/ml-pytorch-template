"""Block definition of resnet model"""
from torch import nn


class Block(nn.Module):
    """Block definition of resnet model"""
    def __init__(self, in_channels, out_channels, identity_down_sample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_down_sample = identity_down_sample

    def forward(self, x):
        """Forward pass"""
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_down_sample(identity)
        x += identity
        x = self.relu(x)
        return x
