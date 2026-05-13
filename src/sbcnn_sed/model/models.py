import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SBCNNSed(nn.Module):
    """
    Pytorch implementation of the SB CNN architecture
    found in the DCASE-models
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # convolution layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        # conv layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
        )

        # conv layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=576, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forwards pass
        Input is x of shape (batch, sequence_frames, n_mels)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.classifier(x)

        return out
