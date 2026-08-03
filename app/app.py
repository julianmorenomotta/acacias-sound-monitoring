import gradio as gr
from inference import detector
import api


def process_audio(audio_filepath):
    if detector is None:
        return {"error": "Model failed to load"}
    if not audio_filepath:
        return {"error": "Audio not provided"}
    return detector.predict(audio_filepath)


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
    demo.app.include_router(api.router)
    demo.launch(server_name="0.0.0.0", share=False)

__all__ = ["detector", "demo", "process_audio"]
