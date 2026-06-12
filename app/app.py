import sys
import importlib
from pathlib import Path

import gradio as gr

current_dir = Path(__file__).resolve().parent
project_root = current_dir if (current_dir / "configs").exists() else current_dir.parent
sys.path.insert(0, str(project_root / "src"))

SoundEventDetector = importlib.import_module(
    "sbcnn_sed.pipeline.inference"
).SoundEventDetector

config_path = project_root / "configs" / "inference.yaml"

try:
    detector = SoundEventDetector(config_path)
    print("Sound event detector model loaded successfully.")
except Exception as e:
    print(f"Failed to load detector model: {e}")
    detector = None


def process_audio(audio_filepath):
    if detector is None:
        return {"error": "Model failed to load."}
    if not audio_filepath:
        return {"error": "Audio not provided."}

    predictions = detector.predict(audio_filepath)
    return predictions


with gr.Blocks(title="Sound Monitor") as demo:
    gr.Markdown("Acacias sound event detection")
    gr.Markdown(
        "Upload an urban soundscape or record from your microphone to detect sound events."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Input Audio")
            submit_btn = gr.Button("Detect Sounds", variant="primary")

        with gr.Column():
            json_output = gr.JSON(label="Detected Events")

    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=json_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
