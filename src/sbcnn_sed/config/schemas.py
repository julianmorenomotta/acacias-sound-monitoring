from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from omegaconf import OmegaConf

from sbcnn_sed.model.models import DEFAULT_CONV_BLOCKS, ConvBlockConfig


class Fold(str, Enum):
    train = "train"
    validate = "validate"
    test = "test"


class MetricType(str, Enum):
    segment_based = "segment_based"
    event_based = "event_based"


@dataclass
class WandBConfig:
    """Weights & Biases tracking settings (from `scripts.train.py`)."""

    entity: str = "julianmrn5-brl-media"
    project: str = "acacias-sound-monitor"


@dataclass
class ModelConfig:
    """Model architecture as data (condumed by hydra.util.instantiate)"""

    _target_: str = "sbcnn_sed.model.models.SBCNNSed"
    num_classes: int = 10
    in_channels: int = 1
    input_shape: tuple[int, int] = (32, 64)
    conv_blocks: list[ConvBlockConfig] = field(
        default_factory=lambda: list(DEFAULT_CONV_BLOCKS)
    )
    fc_hidden: int = 64
    dropout: float = 0.5
    pool_size: int = 2


@dataclass
class DataConfig:
    """Dataset paths, fold and the class list"""

    raw_path: str = "data/raw/URBAN-SED_v2.0.0"
    processed_path: str = "data/processed/URBAN-SED_v2.0.0"
    fold: Fold = Fold.test
    scaler_path: str = "data/processed/URBAN-SED_v2.0.0/scaler.pt"
    classes: list[str] = field(
        default_factory=lambda: [
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
    )


@dataclass
class FeaturesConfig:
    """Mel-spectogram extraction parameters"""

    sample_rate: int = 22050
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 680
    n_mels: int = 64
    sequence_time: float = 1.0
    sequence_hop_time: float = 0.5
    pad_mode: str = "reflect"


@dataclass
class TrainConfig:
    """Training only setting"""

    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 500
    patience: int = 50
    num_workers: int = 4
    seed: int = 42
    checkpoint_path: str = "models/checkpoints/best_sed_model.pth"
    wandb: WandBConfig = field(default_factory=WandBConfig)


@dataclass
class EvalConfig:
    """Evaluation protocol settings.

    NOTE: ``evaluate.py`` uses threshold 0.3 while ``inference.yaml`` uses 0.5;
    the schema makes this inconsistency explicit so we can resolve it.
    """

    threshold: float = 0.3
    merge_gap_seconds: float = 1.0
    metric_type: MetricType = MetricType.segment_based
    report_dir: str = "logs/reports"


@dataclass
class AppConfig:
    """Top-level schema composing every config group."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


__all__ = [
    "AppConfig",
    "DataConfig",
    "EvalConfig",
    "FeaturesConfig",
    "ModelConfig",
    "TrainConfig",
    "WandBConfig",
]


# ---------------------------------------------------------------------------
# Runnable demo:  python -m sbcnn_sed.config.schemas
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Show how the schema behaves: defaults, merging YAML, and validation."""

    print("=" * 70)
    print("1) Resolved config from schema defaults")
    print("=" * 70)
    cfg = OmegaConf.structured(AppConfig())
    print(OmegaConf.to_yaml(cfg))

    # Forward-looking: merge the Phase C2 group YAMLs into the tree if present.
    group_files = {
        "model": "configs/model/sb_cnn.yaml",
        "data": "configs/data/urban_sed.yaml",
        "features": "configs/features/default.yaml",
        "train": "configs/train/default.yaml",
        "eval": "configs/eval/parity.yaml",
    }
    for key, rel_path in group_files.items():
        path = Path(rel_path)
        if path.exists():
            merged = OmegaConf.merge(getattr(cfg, key), OmegaConf.load(path))
            setattr(cfg, key, merged)
            print(f"merged {rel_path}")

    print("=" * 70)
    print("2) Validation: unknown/typo'd keys are rejected")
    print("=" * 70)
    try:
        OmegaConf.merge(cfg, OmegaConf.create({"train": {"learning_rat": 1e-3}}))
    except Exception as exc:  # demo only: show the error, don't crash
        print(f"  -> {type(exc).__name__}: {exc}")

    print("=" * 70)
    print("3) Validation: wrong types are rejected")
    print("=" * 70)
    try:
        OmegaConf.merge(cfg, OmegaConf.create({"train": {"batch_size": "not-an-int"}}))
    except Exception as exc:
        print(f"  -> {type(exc).__name__}: {exc}")

    print("=" * 70)
    print("4) Validation: Literal fields only accept listed values")
    print("=" * 70)
    try:
        OmegaConf.merge(cfg, OmegaConf.create({"data": {"fold": "validation_typo"}}))
    except Exception as exc:
        print(f"  -> {type(exc).__name__}: {exc}")

    print("=" * 70)
    print("5) A valid override is accepted")
    print("=" * 70)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": {"learning_rate": 3e-4}}))
    print(f"  train.learning_rate = {cfg.train.learning_rate}")


if __name__ == "__main__":
    _demo()
