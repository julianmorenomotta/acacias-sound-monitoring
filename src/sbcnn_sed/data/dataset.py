import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from typing import Optional, Tuple
from sbcnn_sed.data.scaler import MinMaxScaler

logger = logging.getLogger(__name__)

URBAN_SED_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


class UrbanSedDataset(Dataset):
    """
    pytorch datasert wrapper for URBAN-SED
    """

    def __init__(
        self,
        processesd_dataset_path: str | Path,
        fold: str,
        scaler: Optional[MinMaxScaler] = None,
    ):

        self.dataset_path = Path(processesd_dataset_path)
        self.fold = fold
        self.scaler = scaler

        if fold not in ["train", "validate", "test"]:
            raise ValueError(f"Got invalid fold value: {fold}")

        self.features_folder = self.dataset_path / "features" / fold
        self.labels_folder = self.dataset_path / "labels" / fold
        self.feature_files = sorted(list(self.features_folder.glob("*.pt")))

        if not self.feature_files:
            logger.warning(
                f"No feature files found for fold {fold} in {self.features_folder}"
            )

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_path = self.feature_files[idx]
        label_path = self.labels_folder / feature_path.name

        features = torch.load(feature_path, weights_only=True)
        labels = torch.load(label_path, weights_only=True)

        if self.scaler is not None:
            features = self.scaler.transform(features)

        return features, labels
