import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from sbcnn_sed.data.features import MelSpectrogramExtractor
from sbcnn_sed.utils.scaler import MinMaxScaler
from sbcnn_sed.utils.constants import URBAN_SED_CLASSES


def get_event_roll(
    label_path: Path,
    num_sequences: int,
    sequence_hop_time: float,
    sequence_time: float,
    classes: list,
) -> torch.Tensor:
    y = np.zeros((num_sequences, len(classes)), dtype=np.float32)

    if not label_path.exists():
        return torch.from_numpy(y)

    labels_df = pd.read_csv(
        label_path, delimiter="\t", header=None, names=["onset", "offset", "label"]
    )
    for _, row in labels_df.iterrows():
        if row["label"] not in classes:
            continue

        c_idx = classes.index(row["label"])
        for i in range(num_sequences):
            seq_start = i * sequence_hop_time
            seq_end = seq_start + sequence_time
            if row["onset"] < seq_end and row["offset"] > seq_start:
                y[i, c_idx] = 1.0

    return torch.from_numpy(y)


def main():
    raw_ds_path = Path("../data/raw/URBAN-SED_v2.0.0")
    processed_ds_path = Path("../data/processed/URBAN-SED_v2.0.0")

    extractor = MelSpectrogramExtractor()
    scaler = MinMaxScaler()

    folds = ["train", "validate", "test"]

    for fold in folds:
        print(f"Processing fold: {fold}")
        audio_dir = raw_ds_path / "audio" / fold
        annot_dir = raw_ds_path / "annotations" / fold
        out_feature_dir = processed_ds_path / "features" / fold
        out_labels_dir = processed_ds_path / "labels" / fold

        out_feature_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        audio_files = list(audio_dir.glob("*.wav"))
        for audio_path in tqdm(audio_files):
            if audio_path.name.startswith("."):
                continue

            # extract features
            features = extractor.extract(audio_path)

            # extract labels
            label_path = annot_dir / (audio_path.stem + ".txt")
            labels = get_event_roll(
                label_path,
                features.shape[0],
                extractor.sequence_hop_time,
                extractor.sequence_time,
                URBAN_SED_CLASSES,
            )

            torch.save(features, out_feature_dir / f"{audio_path.stem}.pt")
            torch.save(labels, out_labels_dir / f"{audio_path.stem}.pt")

            if fold == "train":
                batch_min = torch.min(features).item()
                batch_max = torch.max(features).item()
                if batch_min < scaler.min_val:
                    scaler.min_val = batch_min
                if batch_max > scaler.max_val:
                    scaler.max_val = batch_max

    scaler_path = processed_ds_path / "scaler.pt"
    scaler.save(scaler_path)

    print(f"Preprocessing complete. Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()
