import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import tqdm
import logging
import json
from datetime import datetime

# Ensure src is in the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sbcnn_sed.data.dataset import UrbanSedDataset, URBAN_SED_CLASSES
from sbcnn_sed.data.scaler import MinMaxScaler
from sbcnn_sed.model.models import SBCNNSed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate")

PROCESSED_DATA_PATH = Path("data/processed/URBAN-SED_v2.0.0")
CHECKPOINT_PATH = Path("models/checkpoints/best_sed_model.pth")
BATCH_SIZE = 32
NUM_WORKERS = 4
THRESHOLD = 0.3

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

logger.info(f"Using device: {device}")


def main():
    scaler = MinMaxScaler()
    scaler.load(PROCESSED_DATA_PATH / "scaler.pt")

    test_dataset = UrbanSedDataset(PROCESSED_DATA_PATH, fold="test", scaler=scaler)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    logger.info(f"test batches: {len(test_loader)}")

    model = SBCNNSed(num_classes=10).to(device)

    if not CHECKPOINT_PATH.exists():
        logger.warning(f"Error: checkpoint not found at {CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        f"Loaded checkpoitn from epoch{checkpoint['epoch']} with val loss: {checkpoint.get('best_val_loss', 'N/A')}"
    )

    model.eval()

    total_tp = torch.zeros(10).to(device)
    total_fp = torch.zeros(10).to(device)
    total_fn = torch.zeros(10).to(device)

    logger.info("\n Starting evaluation pass")
    test_loop = tqdm.tqdm(test_loader, desc="testing")

    with torch.no_grad():
        for features, labels in test_loop:
            features, labels = features.to(device), labels.to(device)

            # resahpe
            batch_size, num_sequences, height, width = features.shape
            features = features.view(batch_size * num_sequences, 1, height, width)
            labels = labels.view(batch_size * num_sequences, -1)

            # forward pass
            probs = model.predict(features)

            predictions = (probs > THRESHOLD).float()

            tp = (predictions * labels).sum(dim=0)
            fp = (predictions * (1 - labels)).sum(dim=0)
            fn = ((1 - predictions) * labels).sum(dim=0)

            total_tp += tp
            total_fp += fp
            total_fn += fn

    # calculte segment based metrics
    micro_tp = total_tp.sum().item()
    micro_fp = total_fp.sum().item()
    micro_fn = total_fn.sum().item()

    # micro metrics calculations
    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-8)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-8)
    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
    )

    # macro metrics calculations
    class_precision = total_tp / (total_tp + total_fp + 1e-8)
    class_recall = total_tp / (total_tp + total_fn + 1e-8)
    class_f1 = (
        2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)
    )
    macro_f1 = class_f1.mean().item()

    # error rate calculation
    error_rate = (micro_fp + micro_fn) / (micro_tp + micro_fn + 1e-8)

    # --- 6. Formatting, Logging and Saving ---
    # Construct a dictionary report
    dataset_classes = URBAN_SED_CLASSES

    per_class_results = {}
    for i, cls_name in enumerate(dataset_classes):
        per_class_results[cls_name] = {
            "f1_score": round(class_f1[i].item(), 4),
            "precision": round(class_precision[i].item(), 4),
            "recall": round(class_recall[i].item(), 4),
        }

    results_report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(CHECKPOINT_PATH),
            "threshold": THRESHOLD,
            "trained_epochs": checkpoint.get("epoch", "Unknown"),
        },
        "overall_metrics": {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "error_rate": round(error_rate, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
        },
        "per_class_metrics": per_class_results,
    }

    # Save to JSON
    reports_dir = Path("logs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

    report_file = reports_dir / f"{time_stamp}_evaluation_results.json"

    with open(report_file, "w") as f:
        json.dump(results_report, f, indent=4)

    # logger output
    logger.info("=" * 50)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Model: {CHECKPOINT_PATH.name}")
    logger.info(f"Threshold: {THRESHOLD}")
    logger.info("-" * 50)
    logger.info(f"Macro F1-Score: {macro_f1:.4f}")
    logger.info(f"Micro F1-Score: {micro_f1:.4f}")
    logger.info(f"Error Rate (ER): {error_rate:.4f}")
    logger.info("-" * 50)
    logger.info("Per-Class F1 Scores:")
    for cls_name, metrics in per_class_results.items():
        logger.info(f"  - {cls_name:<15}: {metrics['f1_score']:.4f}")
    logger.info("=" * 50)
    logger.info(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
