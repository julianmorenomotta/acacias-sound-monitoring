import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MinMaxScaler:
    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def fit(self, data_loader):
        """
        Fits the scaler by iterating thru the DataLoader to find the
        global min and mx values in the dataset

        Args:
            data_loader: A Pytorch DataLoader yielding batches of features
        """

        logger.info("Fitting scaler...")

        for batch in data_loader:
            features, _ = batch

            batch_min = torch.min(features)
            batch_max = torch.max(features)

            if batch_min < self.min_val:
                self.min_val = batch_min

            if batch_max > self.max_val:
                self.max_val = batch_max

        logger.info(f"Scaler fitted min: {self.min_val:.4f}, max: {self.max_val:.4f}")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Applies the min-max scaling to a tensor

        Args:
            X (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Scaled tensor in range [-1, 1]
        """
        if self.max_val == self.min_val:
            return torch.zeros_like(X)
        return 2 * ((X - self.min_val) / (self.max_val - self.min_val)) - 1

    def inverse_transform(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """
        Reverts scaling tranformation

        Args:
            X_scaled (torch.Tensor): Scaled tensor

        Returns:
            torch.Tensor: Tensor scaled back to original scale
        """

        return (self.max_val - self.min_val) * ((X_scaled + 1.0) / 2.0) + self.min_val

    def save(self, filepath: str | Path):
        """
        Saves scaler state (min and max values) to file
        """

        state = {"min_val": self.min_val, "max_val": self.max_val}
        torch.save(state, filepath)
        logger.info(f"Scaler state saved to {filepath}")

    def load(self, filepath: str | Path):
        """
        loads the scaler's state from a file
        """
        state = torch.load(filepath)
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
