import torch
import torch.nn as nn
import torch.optim as optim
from sbcnn_sed.model.models import SBCNNSed


class TestTrainingSmoke:
    """Smoke test for the training loop."""

    def test_training_loop_forward_backward(self):
        """
        GIVEN synthetic features/labels and a freshly initialized model
        WHEN we run forward + backward + optimizer.step for 2 steps
        THEN the loss decreases and gradients flow (no crash)

        This tests the core training mechanics without needing the
        URBAN-SED dataset on disk:
          - Model forward pass with BCEWithLogitsLoss
          - Backward pass (gradients flow)
          - Optimizer step (parameters update)
        """
        torch.manual_seed(42)

        num_classes = 10
        batch_size = 4
        # Model expects input of shape (batch, 32, 64) after unsqueeze
        input_shape = (32, 64)
        learning_rate = 1e-3

        model = SBCNNSed(num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create synthetic data — random features and random binary labels
        x = torch.randn(batch_size, *input_shape)
        y = torch.randint(0, 2, (batch_size, num_classes)).float()

        # Step 1 — forward + backward
        logits = model(x)
        loss1 = criterion(logits, y)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step 2 — forward + backward
        logits = model(x)
        loss2 = criterion(logits, y)
        loss2.backward()
        optimizer.step()

        # Loss should decrease after one optimizer step (model learns)
        # Note: on random data this isn't guaranteed, but with a large
        # enough LR and simple data it usually does
        assert loss2 < loss1, (
            f"Loss did not decrease: {loss1.item():.4f} -> {loss2.item():.4f}. "
            "This may indicate a gradient flow issue."
        )

        # Verify gradients were computed (non-None)
        has_grads = all(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert (
            has_grads
        ), "Some parameters have no gradient — backward pass may be broken"
