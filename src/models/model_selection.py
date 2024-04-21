"""Module containing model selection wrapping"""
from torchvision.models import resnet50
from torch import nn

from models.resnet.resnet_18 import ResNet18


class ModelSelection:
    """Class containing model selection wrapping"""
    @staticmethod
    def resnet50_torchvision(n_classes: int, pretrained: bool = True):
        """
        :param n_classes: number of classes
        :param pretrained - specifies if class has to be pretrained or not
        :rtype: nn.Module
        """
        model = nn.Sequential(
            resnet50(pretrained=pretrained),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=n_classes),
        )
        return model

    @staticmethod
    def resnet18_custom_model(num_classes: int, image_channels: int = 3):
        """
        :param num_classes:
        :param image_channels:
        :return: nn.Module - custom resnet model
        """
        return ResNet18(image_channels=image_channels, num_classes=num_classes)
