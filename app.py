import os
from tempfile import NamedTemporaryFile
from typing import Tuple
from zipfile import ZipFile

import gradio as gr
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

from odcnn import ODCNN
from youtube import youtube

accelerator = Accelerator()
device = accelerator.device

DON_MODEL = hf_hub_download(
    repo_id="JacobLinCool/odcnn-320k-100", filename="don_model.pth"
)
KA_MODEL = hf_hub_download(
    repo_id="JacobLinCool/odcnn-320k-100", filename="ka_model.pth"
)


models = {"odcnn-320k-100": ODCNN(DON_MODEL, KA_MODEL, device)}


def run(file: str, model: str, delta: float) -> Tuple[str, str, str]:
    preview, tja = models[model].run(file, delta)

    with NamedTemporaryFile(
        "w", suffix=".tja", delete=True
    ) as tjafile, NamedTemporaryFile("w", suffix=".zip", delete=False) as zfile:
        tjafile.write(tja)

        with ZipFile(zfile.name, "w") as z:
            z.write(file, os.path.basename(file))
            z.write(tjafile.name, f"{os.path.basename(file)}-{model}.tja")

    return preview, tja, zfile.name


def from_youtube(url: str, model: str, delta: float) -> Tuple[str, str, str, str]:
    audio = youtube(url)
    return audio, *run(audio, model, delta)


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

    with gr.Row():
        zip = gr.File(label="Download ZIP", type="filepath")

    with gr.Accordion("Advanced Options", open=False):
        delta = gr.Slider(
            label="Delta",
            value=0.02,
            minimum=0.01,
            maximum=0.5,
            step=0.01,
            info="Threshold for note detection (Ura)",
        )

    btn.click(
        fn=run,
        inputs=[audio, model, delta],
        outputs=[synthesized, tja, zip],
        api_name="run",
    )

    yt_btn.click(
        fn=from_youtube,
        inputs=[yt, model, delta],
        outputs=[audio, synthesized, tja, zip],
    )

app.queue().launch(server_name="0.0.0.0")
