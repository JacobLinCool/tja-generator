import tempfile
from typing import Tuple
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

from model import convNet
from preprocess import Audio, fft_and_melscale
from synthesize import create_tja, detect, synthesize


class ODCNN:
    def __init__(self, don_model: str, ka_model: str, device: torch.device = "cpu"):
        donNet = convNet()
        donNet = donNet.to(device)
        donNet.load_state_dict(torch.load(don_model, map_location="cpu"))
        self.donNet = donNet

        kaNet = convNet()
        kaNet = kaNet.to(device)
        kaNet.load_state_dict(torch.load(ka_model, map_location="cpu"))
        self.kaNet = kaNet

        self.device = device

    def run(self, file: str, delta=0.05) -> Tuple[str, str]:
        data, sr = sf.read(file, always_2d=True)
        song = Audio(data, sr)
        song.data = song.data.mean(axis=1)
        song.feats = fft_and_melscale(
            song,
            nhop=512,
            nffts=[1024, 2048, 4096],
            mel_nband=80,
            mel_freqlo=27.5,
            mel_freqhi=16000.0,
        )

        don_inference = self.donNet.infer(song.feats, self.device, minibatch=4192)
        don_inference = np.reshape(don_inference, (-1))

        ka_inference = self.kaNet.infer(song.feats, self.device, minibatch=4192)
        ka_inference = np.reshape(ka_inference, (-1))

        easy_detection = detect(don_inference, ka_inference, delta=0.25)
        normal_detection = detect(don_inference, ka_inference, delta=0.2)
        hard_detection = detect(don_inference, ka_inference, delta=0.15)
        oni_detection = detect(don_inference, ka_inference, delta=0.075)
        ura_detection = detect(don_inference, ka_inference, delta=delta)

        synthesized_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        synthesize(*hard_detection, song, synthesized_path)
        file = Path(file)
        tja = create_tja(
            song,
            timestamps=[
                easy_detection,
                normal_detection,
                hard_detection,
                oni_detection,
                ura_detection,
            ],
            title=file.stem,
            wave=file.name,
        )

        return synthesized_path, tja
