"""Test ofr the typed configuration layer

Validates:
- the dataclass schemas resolbe to sane defaults
- every YAML goup file in group file in ``configs/`` maps cleany onto its schema
- the primary ``configs/config.yaml``composes the way Hydra entry points will
- invalid overrides (unkown keys, wrong types, bad enum values) are rejected

"""

import torch
import hydra
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from sbcnn_sed.model.models import SBCNNSed
from sbcnn_sed.config import AppConfig


class Testinstantiate:
    """hydra.util.instantiate(cfg.model) build a real, working model."""

    def test_schema_instantiate_model(self):
        cfg = OmegaConf.structured(AppConfig())
        model = hydra.utils.instantiate(cfg.model)
        assert isinstance(model, SBCNNSed)

    def test_composed_config_instantiates_and_forward(self):
        with hydra.initialize(config_path="../configs", version_base=None):
            composed = hydra.compose(config_name="config")
        model = hydra.utils.instantiate(composed.model)
        x = torch.rand(1, 32, 64)
        out = model(x)
        assert out.shape == (1, composed.model.num_classes)

    def test_parameter_count_matches_baseline(self):
        """the config driven model is bit-identical in size to the original."""
        cfg = OmegaConf.structured(AppConfig())
        model = hydra.utils.instantiate(cfg.model)
        total = sum(p.numel() for p in model.parameters())
        assert total == 244_554

    def test_architecture_is_parameterized(self):
        """Changing conv_blocks in config changes the model — no code edit."""
        cfg = OmegaConf.structured(AppConfig())
        cfg.model.conv_blocks[0].out_channels = 32
        model = hydra.utils.instantiate(cfg.model)
        assert model.encoder[0][0].out_channels == 32


class TestSchemaDefaults:
    """The schema is the source of truth: defaults resolve and are typed."""

    def test_resolved_defaults(self):
        cfg = OmegaConf.structured(AppConfig())
        assert cfg.train.batch_size == 32
        assert cfg.train.learning_rate == 1e-4
        assert cfg.train.epochs == 500
        assert cfg.train.patience == 50
        assert cfg.train.seed == 42
        assert cfg.data.fold == "test"
        assert cfg.features.sample_rate == 22050
        assert cfg.features.n_mels == 64
        assert cfg.features.hop_length == 680
        assert cfg.model.num_classes == 10
        assert cfg.eval.threshold == 0.3
        assert cfg.eval.metric_type == "segment_based"

    def test_clases_are_ten_urban_sed_classes(self):
        cfg = OmegaConf.structured(AppConfig())
        assert len(cfg.data.classes) == 10
        assert cfg.data.classes[3] == "dog_bark"
        assert "gun_shot" in cfg.data.classes
        assert "jackhammer" in cfg.data.classes


GROUP_FILES = [
    ("model", "configs/model/sb_cnn.yaml"),
    ("data", "configs/data/urban_sed.yaml"),
    ("features", "configs/features/default.yaml"),
    ("train", "configs/train/default.yaml"),
    ("eval", "configs/eval/parity.yaml"),
]


class TestYamlGroupFiles:
    """Every YAMl group file must map onto its schema with no surprises."""

    @pytest.mark.parametrize("group,path", GROUP_FILES)
    def test_yaml_merge_into_schema(self, group, path):
        schema = OmegaConf.structured(AppConfig())
        yaml_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(getattr(schema, group), yaml_cfg)
        assert merged is not None

    @pytest.mark.parametrize("group,path", GROUP_FILES)
    def test_yaml_values_survive_merge(self, group, path):
        schema = OmegaConf.structured(AppConfig())
        yaml_cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(getattr(schema, group), yaml_cfg)
        for key, value in yaml_cfg.items():
            assert merged[key] == value, f"{group}.{key}"


class TestPrimaryConfig:
    """The primary config composes the groups like a Hydra entry point will."""

    def test_congif_yam_composes(self):
        with hydra.initialize(config_path="../configs", version_base=None):
            composed = hydra.compose(config_name="config")

        validated = OmegaConf.merge(OmegaConf.structured(AppConfig()), composed)
        assert validated.train.batch_size == 32
        assert validated.data.fold == "test"
        assert len(validated.data.classes) == 10


class TestValidation:
    """Invalid input is rejected at load time, not discovered at runtime."""

    def test_unknown_key_rejected(self):
        cfg = OmegaConf.structured(AppConfig())
        with pytest.raises(ConfigKeyError):
            OmegaConf.merge(cfg, OmegaConf.create({"train": {"learning_rae": 1e-3}}))

    def test_wrong_type_rejected(self):
        cfg = OmegaConf.structured(AppConfig())
        with pytest.raises(ValidationError):
            OmegaConf.merge(cfg, OmegaConf.create({"train": {"batch_size": "none"}}))

    def test_invalid_fold_rejected(self):
        cfg = OmegaConf.structured(AppConfig())
        with pytest.raises(ValidationError):
            OmegaConf.merge(cfg, OmegaConf.create({"data": {"fold": "nope"}}))

    def test_invalid_metric_type(self):
        cfg = OmegaConf.structured(AppConfig())
        with pytest.raises(ValidationError):
            OmegaConf.merge(
                cfg, OmegaConf.create({"eval": {"metric_type": "frame_based"}})
            )
