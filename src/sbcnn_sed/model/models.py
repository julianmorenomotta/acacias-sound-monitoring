import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass()
class ConvBlockConfig:
    """Parameters for on conv block of the SB-CNN encoder."""

    out_channels: int
    kernel_size: int = 5
    pool: bool = True


# Default architecture: matching DCASE SB_CNN_SED
DEFAULT_CONV_BLOCKS: tuple[ConvBlockConfig, ...] = (
    ConvBlockConfig(out_channels=64, kernel_size=5, pool=True),
    ConvBlockConfig(out_channels=64, kernel_size=5, pool=True),
    ConvBlockConfig(out_channels=64, kernel_size=5, pool=False),
)


class SBCNNSed(nn.Module):
    """
    Pytorch implementation of the SB CNN architecture
    found in the DCASE-models
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        input_shape: Sequence[int] = (32, 64),
        conv_blocks: Sequence[ConvBlockConfig] = DEFAULT_CONV_BLOCKS,
        fc_hidden: int = 64,
        dropout: float = 0.5,
        pool_size: int = 2,
    ):
        super().__init__()

        self.num_classes = num_classes

        blocks = [
            ConvBlockConfig(
                out_channels=int(b.out_channels),
                kernel_size=int(b.kernel_size),
                pool=bool(b.pool),
            )
            for b in conv_blocks
        ]

        layers: list[nn.Module] = []
        cin = in_channels
        for block in blocks:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        cin,
                        block.out_channels,
                        kernel_size=block.kernel_size,
                        padding="valid",
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(pool_size, pool_size) if block.pool else nn.Identity(),
                    nn.BatchNorm2d(block.out_channels),
                )
            )
            cin = block.out_channels
        self.encoder = nn.Sequential(*layers)

        # Compute the classifier's input size from the actual conv output
        h, w = int(input_shape[0]), int(input_shape[1])
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.encoder(dummy)
            n_features = out.shape[1:].numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(n_features, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (batch, sequence_frames, n_mels) or (batch, 1, H, W)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
