import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel


class Audio:
    """
    audio class which holds music data and timestamp for notes.

    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.
    Variables:


    Example:
        >>>from music_processor import *
        >>>song = Audio(filename)
        >>># to get audio data
        >>>song.data
        >>># to import .tja files:
        >>>song.import_tja(filename)
        >>># to get data converted
        >>>song.data = (song.data[:,0]+song.data[:,1])/2
        >>>fft_and_melscale(song, include_zero_cross=False)
    """

    def __init__(self, data, samplerate, stereo=True):
        self.data = data
        self.samplerate = samplerate
        if stereo is False:
            self.data = (self.data[:, 0] + self.data[:, 1]) / 2
        self.timestamp = []

    def plotaudio(self, start_t, stop_t):

        plt.plot(
            np.linspace(start_t, stop_t, stop_t - start_t), self.data[start_t:stop_t, 0]
        )
        plt.show()

    def save(self, filename, start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)

    def synthesize(self, diff=True, don="./asset/don.wav", ka="./asset/ka.wav"):

        donsound = sf.read(don)[0]
        donsound = (donsound[:, 0] + donsound[:, 1]) / 2
        kasound = sf.read(ka)[0]
        kasound = (kasound[:, 0] + kasound[:, 1]) / 2
        donlen = len(donsound)
        kalen = len(kasound)

        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0] * self.samplerate)
                try:
                    if stamp[1] in (1, 3, 5, 6, 7):
                        self.data[timing : timing + donlen] += donsound
                    elif stamp[1] in (2, 4):
                        self.data[timing : timing + kalen] += kasound
                except ValueError:
                    pass

        elif diff == "don":
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp * self.samplerate + donlen < self.data.shape[0]:
                        self.data[
                            int(stamp[0] * self.samplerate) : int(
                                stamp[0] * self.samplerate
                            )
                            + donlen
                        ] += donsound
            else:
                for stamp in self.timestamp:
                    if stamp * self.samplerate + donlen < self.data.shape[0]:
                        self.data[
                            int(stamp * self.samplerate) : int(stamp * self.samplerate)
                            + donlen
                        ] += donsound

        elif diff == "ka":
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp * self.samplerate + kalen < self.data.shape[0]:
                        self.data[
                            int(stamp[0] * self.samplerate) : int(
                                stamp[0] * self.samplerate
                            )
                            + kalen
                        ] += kasound
            else:
                for stamp in self.timestamp:
                    if stamp * self.samplerate + kalen < self.data.shape[0]:
                        self.data[
                            int(stamp * self.samplerate) : int(stamp * self.samplerate)
                            + kalen
                        ] += kasound


def make_frame(data, nhop, nfft):
    """
    helping function for fftandmelscale.
    細かい時間に切り分けたものを学習データとするため，nhop(512)ずつずらしながらnfftサイズのデータを配列として返す
    """

    length = data.shape[0]
    framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
    return np.array(
        [framedata[i * nhop : i * nhop + nfft] for i in range(length // nhop)]
    )


# @jit
def fft_and_melscale(
    song,
    nhop=512,
    nffts=[1024, 2048, 4096],
    mel_nband=80,
    mel_freqlo=27.5,
    mel_freqhi=16000.0,
    include_zero_cross=False,
):
    """
    fft and melscale method.
    fft: nfft = [1024, 2048, 4096]; サンプルの切り取る長さを変えながらデータからnp.arrayを抽出して高速フーリエ変換を行う．
    melscale: 周波数の次元を削減するとともに，log10の値を取っている．
    """

    feat_channels = []

    for nfft in nffts:

        feats = []
        window = signal.windows.blackmanharris(nfft)
        filt = mel(
            sr=song.samplerate,
            n_fft=nfft,
            n_mels=mel_nband,
            fmin=mel_freqlo,
            fmax=mel_freqhi,
        )

        # get normal frame
        frame = make_frame(song.data, nhop, nfft)
        # print(frame.shape)

        # melscaling
        processedframe = fft(window * frame)[:, : nfft // 2 + 1]
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe) ** 2))
        processedframe = 20 * np.log10(processedframe + 0.1)
        # print(processedframe.shape)

        feat_channels.append(processedframe)

    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        print(song.zero_crossing)

    return np.array(feat_channels)
