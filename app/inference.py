import sys
import importlib
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir if (current_dir / "configs").exists() else current_dir.parent
sys.path.insert(0, str(project_root / "src"))

SoundEventDetector = importlib.import_module(
    "sbcnn_sed.pipeline.inference"
).SoundEventDetector

config_path = project_root / "configs" / "inference.yaml"
try:
    detector = SoundEventDetector(config_path)
    print("Model succesfully loaded.")

except Exception as e:
    print(f"Failed to load detector model: {e}")
    detector = None
