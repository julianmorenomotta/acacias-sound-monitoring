import yaml
import torch
import numpy as np
from pathlib import Path

from sbcnn_sed.utils.constants import URBAN_SED_CLASSES
from sbcnn_sed.utils.scaler import MinMaxScaler
from sbcnn_sed.model.models import SBCNNSed
from sbcnn_sed.data.features import MelSpectrogramExtractor


class SoundEventDetector:
    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        config_dir = config_path.parent

        for key, path in self.config.get("paths", {}).items():
            p = Path(path)
            if not p.is_absolute():
                self.config["paths"][key] = str((config_dir / p).resolve())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_cfg = self.config.get("audio", {})
        self.extractor = MelSpectrogramExtractor(
            sample_rate=audio_cfg.get("sample_rate", 22050),
            sequence_time=audio_cfg.get("sequence_time", 1.0),
            sequence_hop_time=audio_cfg.get("sequence_hop_time", 0.5),
        )

        # load the scaler
        scaler_path = self.config["paths"]["scaler"]
        self.scaler = MinMaxScaler()
        self.scaler.load(scaler_path)

        # model initialization
        self.model = SBCNNSed()
        weights_path = self.config["paths"]["model_weights"]

        checkpoint = torch.load(weights_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path: str) -> list[dict]:
        """
        Main public method to predict sound events from an audio file.
        """
        # extract mel spectrograms
        features = self.extractor.extract(audio_path)

        # scale the features
        np_features = features.numpy()
        scaled_features = self.scaler.transform(np_features)

        inputs = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(1)
        inputs = inputs.to(self.device)

        # Forward pass
        with torch.no_grad():
            probs = self.model.predict(inputs)
            probs = probs.cpu().numpy()
        # Convert output to numpy and smooth the events
        # return the JSON event list
        return self._smooth_events(probs)

    def _smooth_events(self, probabilities: np.ndarray) -> list[dict]:
        """
        Input: a frame by frame probability matrix.
        Returns a lis of structured events.
        """
        inf_cfg = self.config.get("inference", {})
        threshold = inf_cfg.get("confidence_threshold", 0.5)
        merge_gap = inf_cfg.get("merge_gap_seconds", 1.0)

        audio_cfg = self.config.get("audio", {})
        hop_time = audio_cfg.get("sequence_hop_time", 0.5)
        seq_time = audio_cfg.get("sequence_time", 1.0)

        events = []
        num_windows, num_classes = probabilities.shape

        for class_idx in range(num_classes):
            class_name = URBAN_SED_CLASSES[class_idx]
            class_probs = probabilities[:, class_idx]

            # find which windows crossed the threshold for the class
            active_frames = np.where(class_probs > threshold)[0]
            if len(active_frames) == 0:
                continue

            current_event = {
                "event": class_name,
                "start": float(active_frames[0] * hop_time),
                "end": float(active_frames[0] * hop_time + seq_time),
                "confidence": float(class_probs[active_frames[0]]),
            }

            for i in range(1, len(active_frames)):
                frame_idx = active_frames[i]
                frame_start = float(frame_idx * hop_time)
                frame_end = float(frame_start + seq_time)
                frame_prob = float(class_probs[frame_idx])

                if (frame_start - current_event["end"]) <= merge_gap:
                    current_event["end"] = max(current_event["end"], frame_end)
                    current_event["confidence"] = max(
                        current_event["confidence"], frame_prob
                    )

                else:
                    events.append(current_event)
                    current_event = {
                        "event": class_name,
                        "start": frame_start,
                        "end": frame_end,
                        "confidence": frame_prob,
                    }

            events.append(current_event)
        events.sort(key=lambda x: x["start"])

        return events
