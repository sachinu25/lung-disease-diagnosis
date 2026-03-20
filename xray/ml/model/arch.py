import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        """
        Creating custom CNN architecture for Image classification
        """
        super(Net, self).__init__()

        self.convolution_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.pooling11 = nn.MaxPool2d(2, 2)

        self.convolution_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8, out_channels=20, kernel_size=(3, 3), padding=0, bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        self.pooling22 = nn.MaxPool2d(2, 2)

        self.convolution_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.pooling33 = nn.MaxPool2d(2, 2)

        self.convolution_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(20),
        )

        self.convolution_block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.convolution_block6 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.convolution_block7 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(10),
        )

        self.convolution_block8 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=14,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(14),
        )

        self.convolution_block9 = nn.Sequential(
            nn.Conv2d(
                in_channels=14,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))

        self.convolution_block_out = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=(4, 4), padding=0, bias=True
            ),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.convolution_block1(x)

        x = self.pooling11(x)

        x = self.convolution_block2(x)

        x = self.pooling22(x)

        x = self.convolution_block3(x)

        x = self.pooling33(x)

        x = self.convolution_block4(x)

        x = self.convolution_block5(x)

        x = self.convolution_block6(x)

        x = self.convolution_block7(x)

        x = self.convolution_block8(x)

        x = self.convolution_block9(x)

        x = self.gap(x)

        x = self.convolution_block_out(x)

        x = x.view(-1, 2)   # reshape to 1D tensor for classification

        return x  # return logits for CrossEntropyLoss


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 with ImageNet pre-training for pneumonia classification."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(EfficientNetB0, self).__init__()
        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all feature layers for head-only training."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, num_blocks: int = 3) -> None:
        """Unfreeze the last *num_blocks* feature blocks for fine-tuning."""
        feature_blocks = list(self.backbone.features.children())
        for block in feature_blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


class ResNet18(nn.Module):
    """ResNet-18 with ImageNet pre-training for pneumonia classification."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(ResNet18, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except the final fully-connected head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_layers(self, num_blocks: int = 2) -> None:
        """Unfreeze the last *num_blocks* residual layer groups for fine-tuning."""
        layer_names = ["layer4", "layer3", "layer2", "layer1"]
        for name, param in self.backbone.named_parameters():
            for layer_name in layer_names[:num_blocks]:
                if name.startswith(layer_name):
                    param.requires_grad = True


def get_model(
    model_type: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Factory function to select and return the requested model.

    Args:
        model_type: One of ``"efficientnet_b0"``, ``"resnet18"``, or
            ``"custom"`` for the built-in CNN.
        num_classes: Number of output classes (default 2 for binary).
        pretrained: Use ImageNet pre-trained weights for transfer learning
            models.

    Returns:
        The requested :class:`torch.nn.Module`.
    """
    if model_type == "efficientnet_b0":
        return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "custom":
        return Net()
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from 'efficientnet_b0', 'resnet18', or 'custom'."
        )
