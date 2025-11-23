# train/mobilenet_b0.py

import torch.nn as nn
from torchvision import models


def create_mobilenet_b0(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Rebuild the same MobileNetV2 architecture used in Colab:
    - pretrained on ImageNet
    - final classifier layer replaced for 10 classes
    """
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.mobilenet_v2(weights=weights)

    # Replace final layer to match the number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
