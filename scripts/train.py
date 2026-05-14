import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import sys
import logging
import wandb
import tqdm as tqdm

sys.path.append(str(Path(__file__).parent.parent / "src"))

from sbcnn_sed.data.dataset import UrbanSedDataset
from sbcnn_sed.data.scaler import MinMaxScaler
from sbcnn_sed.model.models import SBCNNSed

PROCESSED_DATA_PATH = Path("data/processed/URBAN-SED_v2.0.0")
MODEL_SAVE_PATH = Path("models/checkpoints/best_sed_model.pth")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 1
PATIENCE = 10
NUM_WORKERS = 4

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

wandb.init(
    entity="julianmrn5-brl-media",
    project="acacias-sound-monitor",
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
    },
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# load the scaler
scaler = MinMaxScaler()
scaler.load(PROCESSED_DATA_PATH / "scaler.pt")

# datasets
train_dataset = UrbanSedDataset(PROCESSED_DATA_PATH, fold="train", scaler=scaler)
val_dataset = UrbanSedDataset(PROCESSED_DATA_PATH, fold="validate", scaler=scaler)

# dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

logger.info(f"Train batches: {len(train_dataset)} | Val batches: {len(val_loader)}")

model = SBCNNSed(num_classes=10).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

wandb.watch(model, log_freq=10)
logger.info(f"starting training on device: {device}")

# main training loop
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    for features, labels in train_loop:
        features, labels = features.to(device), labels.to(device)

        batch_size, num_sequences, height, width = features.shape
        features = features.view(batch_size * num_sequences, 1, height, width)
        labels = labels.view(batch_size * num_sequences, -1)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # validate
    model.eval()
    val_loss = 0.0

    val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
    with torch.no_grad():
        for features, labels in val_loop:
            features, labels = features.to(device), labels.to(device)

            batch_size, num_sequences, height, width = features.shape
            features = features.view(batch_size * num_sequences, 1, height, width)
            labels = labels.view(batch_size * num_sequences, -1)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Log to CLI, File, and W&B
    log_msg = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    logger.info(log_msg)
    print(f"\n{log_msg}")

    wandb.log(
        {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch + 1}
    )

    # set up early stopping and checkpoint saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        best_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }

        torch.save(best_checkpoint, MODEL_SAVE_PATH)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        interval_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }

        interval_path = MODEL_SAVE_PATH.parent / f"checkpoint_epoch{epoch+1}.pth"
        torch.save(interval_checkpoint, interval_path)

    if epochs_without_improvement >= PATIENCE:
        msg = f"Early stopping triggered after {epoch+1} epochs."
        logger.info(msg)
        print(f"\n{msg}")
        break

print("training complete")
wandb.finish()
