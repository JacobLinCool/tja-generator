import gradio as gr
from typing import Tuple
import numpy as np
import soundfile as sf
import tempfile
from model import *
from preprocess import *
from synthesize import *

DON_MODEL = "./models/don_model.pth"
KA_MODEL = "./models/ka_model.pth"

useCUDA = torch.cuda.is_available()
device = torch.device("cuda" if useCUDA else "cpu")

donNet = convNet()
donNet = donNet.to(device)
donNet.load_state_dict(torch.load(DON_MODEL, map_location=device))

kaNet = convNet()
kaNet = kaNet.to(device)
kaNet.load_state_dict(torch.load(KA_MODEL, map_location=device))

def run(file: str) -> Tuple[str, str]:
    data, sr = sf.read(file, always_2d=True)
    song = Audio(data, sr)
    song.data = (song.data[:, 0] + song.data[:, 1]) / 2
    song.feats = fft_and_melscale(
        song,
        nhop=512,
        nffts=[1024, 2048, 4096],
        mel_nband=80,
        mel_freqlo=27.5,
        mel_freqhi=16000.0,
    )

    don_inference = donNet.infer(song.feats, device, minibatch=4192)
    don_inference = np.reshape(don_inference, (-1))

    ka_inference = kaNet.infer(song.feats, device, minibatch=4192)
    ka_inference = np.reshape(ka_inference, (-1))

    synthesized_path = tempfile.NamedTemporaryFile(suffix=".wav").name
    detection(don_inference, ka_inference, song, synthesized_path)
    tja = create_tja(song, song.don_timestamp, song.ka_timestamp)

    return synthesized_path, tja



app = gr.Interface(
    fn=run,
    inputs=[gr.Audio(label="Music", type="filepath")],
    outputs=[gr.Audio(label="Synthesized Audio"), gr.Text(label="TJA")],
)
app.queue().launch()
