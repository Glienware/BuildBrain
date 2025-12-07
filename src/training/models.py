"""
Deep Learning Models

Pre-trained models for computer vision tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
from torchvision.models import VGG16_Weights, VGG19_Weights
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B4_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import MobileNet_V2_Weights


class ResNet18(nn.Module):
    """ResNet-18 model for image classification."""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    """ResNet-50 model for image classification."""

    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    """ResNet-101 model for image classification."""

    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        # Replace the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    """VGG16 model for image classification."""

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG19(nn.Module):
    """VGG19 model for image classification."""

    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 model for image classification."""

    def __init__(self, num_classes=10):
        super(EfficientNetB0, self).__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB4(nn.Module):
    """EfficientNet-B4 model for image classification."""

    def __init__(self, num_classes=10):
        super(EfficientNetB4, self).__init__()
        self.model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    """DenseNet-121 model for image classification."""

    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        # Replace the final layer
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 model for image classification."""

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        # Replace the final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for tabular data."""

    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64]):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_model(model_name, num_classes=10, input_size=None):
    """
    Factory function to get the appropriate model.

    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        input_size: Input size for tabular models (MLP)

    Returns:
        PyTorch model instance
    """
    if model_name == "ResNet-18":
        return ResNet18(num_classes)
    elif model_name == "ResNet-50":
        return ResNet50(num_classes)
    elif model_name == "ResNet-101":
        return ResNet101(num_classes)
    elif model_name == "VGG16":
        return VGG16(num_classes)
    elif model_name == "VGG19":
        return VGG19(num_classes)
    elif model_name == "EfficientNet-B0":
        return EfficientNetB0(num_classes)
    elif model_name == "EfficientNet-B4":
        return EfficientNetB4(num_classes)
    elif model_name == "DenseNet-121":
        return DenseNet121(num_classes)
    elif model_name == "MobileNetV2":
        return MobileNetV2(num_classes)
    elif model_name == "PyTorch CNN":
        return SimpleCNN(num_classes)
    elif model_name == "PyTorch MLP":
        if input_size is None:
            raise ValueError("input_size required for MLP model")
        return SimpleMLP(input_size, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")