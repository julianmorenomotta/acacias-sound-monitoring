import torch
import pytest
import tempfile
from pathlib import Path
from sbcnn_sed.utils.scaler import MinMaxScaler


class TestMinMaxScaler:
    """Unit test for MinMaxScaler"""

    def test_fit_and_transform_basic(self):
        """Scale [0, 2, 4] -> min=0, max=4 -> [-1, 0, 1]"""
        scaler = MinMaxScaler()
        scaler.min_val = 0.0
        scaler.max_val = 4.0
        x = torch.tensor([[0.0, 2.0, 4.0]])
        result = scaler.transform(x)
        expected = torch.tensor([[-1.0, 0.0, 1.0]])
        assert torch.allclose(result, expected)

    def test_fit_and_transform_identity(self):
        """[-1, 0, 1] → min=-1, max=1 → unchanged"""
        scaler = MinMaxScaler()
        scaler.min_val = -1.0
        scaler.max_val = 1.0
        x = torch.tensor([[-1.0, 0.0, 1.0]])
        result = scaler.transform(x)
        assert torch.allclose(result, x)

    def test_inverse_transform_roundtrip(self):
        """transform → inverse_transform recovers original"""
        scaler = MinMaxScaler()
        scaler.min_val = 0.0
        scaler.max_val = 10.0
        x = torch.tensor([[0.0, 2.5, 5.0, 7.5, 10.0]])
        scaled = scaler.transform(x)
        restored = scaler.inverse_transform(scaled)
        assert torch.allclose(x, restored, atol=1e-6)

    def test_fit_via_dataloader(self):
        """fit() iterates a  DataLoader to find global min/max"""
        scaler = MinMaxScaler()
        dataset = [
            (torch.tensor([[1.0, 2.0]]), torch.tensor([0])),
            (torch.tensor([[3.0, 8.0]]), torch.tensor([1])),
        ]
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        scaler.fit(loader)
        assert scaler.min_val == 1.0
        assert scaler.max_val == 8.0

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save scaler state to disk and reload it"""
        scaler1 = MinMaxScaler()
        scaler1.min_val = -3.0
        scaler1.max_val = 12.0
        path = tmp_path / "scaler.pt"
        scaler1.save(path)

        scaler2 = MinMaxScaler()
        scaler2.load(path)
        assert scaler2.min_val == -3.0
        assert scaler2.max_val == 12.0

    def test_constan_value_returns_zeros(self):
        """Whne min == max returns all zeros and no crash"""
        scaler = MinMaxScaler()
        scaler.min_val = 5.0
        scaler.max_val = 5.0
        x = torch.tensor([[5.0, 5.0, 5.0]])
        result = scaler.transform(x)
        expected = torch.zeros_like(x)
        assert torch.allclose(result, expected)
