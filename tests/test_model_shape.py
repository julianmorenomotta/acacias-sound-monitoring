import torch
import pytest
from sbcnn_sed.model.models import SBCNNSed


class TestSBCNNSedShapes:
    """Shape and forward pass test for SBCNNSed."""

    MODEL_INPUT_SHAPE = (1, 32, 64)
    EXPECTED_PARAMS = 244_554
    NUM_CLASSES = 10

    def test_parameter_count(self):
        model = SBCNNSed()
        total = sum(p.numel() for p in model.parameters())
        assert (
            total == self.EXPECTED_PARAMS
        ), f"Expected {self.EXPECTED_PARAMS}, got {total}"

    def test_forward_shape_batch1(self):
        model = SBCNNSed()
        x = torch.rand(1, 32, 64)
        out = model(x)
        assert out.shape == (1, self.NUM_CLASSES), f"Got {out.shape}"

    def test_output_is_finite(self):
        model = SBCNNSed()
        x = torch.rand(4, 32, 64)
        out = model(x)
        assert torch.isfinite(out).all(), "Found NaN or Inf in output"

    def test_predict_return_probabilities(self):
        model = SBCNNSed()
        x = torch.randn(4, 32, 64)
        probs = model.predict(x)
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), "Probabilities oout of [0, 1] range"
