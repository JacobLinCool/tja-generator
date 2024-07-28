import os
import gradio as gr
from typing import Tuple
from accelerate import Accelerator
from odcnn import ODCNN
from youtube import youtube

accelerator = Accelerator()
device = accelerator.device

DON_MODEL = "./models/don_model.pth"
KA_MODEL = "./models/ka_model.pth"


models = {"odcnn-320k-100": ODCNN(DON_MODEL, KA_MODEL, device)}


def run(file: str, model: str) -> Tuple[str, str]:
    return models[model].run(file)


def from_youtube(url: str, model: str) -> Tuple[str, str, str]:
    audio = youtube(url)
    return audio, *run(audio, model)


with gr.Blocks() as app:
    with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
        README = f.read()
        # remove yaml front matter
        blocks = README.split("---")
        if len(blocks) > 1:
            README = "---".join(blocks[2:])

    gr.Markdown(README)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload an audio file")
            audio = gr.Audio(label="Upload an audio file", type="filepath")
        with gr.Column():
            gr.Markdown(
                "## or use a YouTube URL\n\nTry something on [The First Take](https://www.youtube.com/@The_FirstTake)?"
            )
            yt = gr.Textbox(
                label="YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )
            yt_btn = gr.Button("Use this YouTube URL")

    with gr.Row():
        model = gr.Radio(
            label="Select a model",
            choices=[s for s in models.keys()],
            value="odcnn-320k-100",
        )
        btn = gr.Button("Infer", variant="primary")

    with gr.Row():
        with gr.Column():
            synthesized = gr.Audio(
                label="Synthesized Audio",
                format="mp3",
                type="filepath",
                interactive=False,
            )
        with gr.Column():
            tja = gr.Text(label="TJA", interactive=False)

    btn.click(
        fn=run,
        inputs=[audio, model],
        outputs=[synthesized, tja],
        api_name="run",
    )

    yt_btn.click(
        fn=from_youtube,
        inputs=[yt, model],
        outputs=[audio, synthesized, tja],
    )

app.queue().launch(server_name="0.0.0.0")
