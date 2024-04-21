"""Resnet 18 model implementation"""
from torch import nn
from models.resnet.block import Block


# pylint: disable=R0902
class ResNet18(nn.Module):
    """Class of ResNet 18 model"""
    def __init__(self, image_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):
        identity_down_sample = None
        if stride != 1:
            identity_down_sample = self.identity_down_sample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_down_sample=identity_down_sample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        """
        Implements forward pass
        :rtype: torch.tensor
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    @staticmethod
    def identity_down_sample(in_channels: int, out_channels: int):
        """
        :param in_channels: count of input channels
        :param out_channels: count of output channels
        :rtype: object
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
