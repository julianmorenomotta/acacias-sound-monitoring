import sys
import gradio as gr
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent
sys.path.append(str(root_dir / "src"))

from sbcnn_sed.pipeline.inference import SoundEventDetector

config_path = root_dir / "configs" / "inference.yaml"

try:
    detector = SoundEventDetector(config_path)
    print("Sound event detecrot model loaded successfully.")
except Exception as e:
    print(f"Faild to load detector model: {e}")
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
