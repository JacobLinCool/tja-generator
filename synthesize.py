from typing import List
import numpy as np
from librosa.util import peak_pick

from preprocess import *


def smooth(x, window_len=11, window="hanning"):
    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y


def detect(don_inference, ka_inference, delta=0.05):
    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (
        peak_pick(
            x=don_inference,
            pre_max=1,
            post_max=2,
            pre_avg=4,
            post_avg=5,
            delta=delta,
            wait=3,
        )
        + 7
    )  # 実際は7フレーム目のところの音
    ka_timestamp = (
        peak_pick(
            x=ka_inference,
            pre_max=1,
            post_max=2,
            pre_avg=4,
            post_avg=5,
            delta=delta,
            wait=3,
        )
        + 7
    )

    print(don_timestamp)
    print(ka_timestamp)

    don_timestamp = don_timestamp[
        np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])
    ]

    ka_timestamp = ka_timestamp[
        np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])
    ]

    return don_timestamp, ka_timestamp


# def note_to_drumroll(timestamp, max_gap=5, min_note=3):
#     drumroll = []
#     note = 0
#     for i in range(1, len(timestamp)):
#         if timestamp[i] - timestamp[i - 1] <= max_gap:
#             note += 1
#         else:
#             if note >= min_note:
#                 drumroll.append((timestamp[i - note - 1], timestamp[i - 1]))
#             note = 0
#     if note >= min_note:
#         drumroll.append((timestamp[-note - 1], timestamp[-1]))
#     return drumroll

def synthesize(don_timestamp, ka_timestamp, song, filepath):
    song.don_timestamp = don_timestamp
    song.timestamp = song.don_timestamp * 512 / song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff="don")

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp
    song.timestamp = song.ka_timestamp * 512 / song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff="ka")

    song.save(filepath)


def create_tja(
    song,
    timestamps: List[tuple],
    title="untitled",
    subtitle="--",
    wave="untitled.ogg",
    safezone=2,
):
    tja = f"TITLE: {title}\nSUBTITLE: {subtitle}\nBPM: 240\nWAVE:{wave}\nOFFSET:0\n\n"

    for i, (don, ka) in enumerate(timestamps):
        try:
            level = [3, 5, 7, 8, 9][i]
            scroll = [0.6, 0.7, 0.8, 0.9, 1.0][i]

            don_timestamp = np.rint(don * 512 / song.samplerate * 100).astype(np.int32)
            ka_timestamp = np.rint(ka * 512 / song.samplerate * 100).astype(np.int32)
            length = np.max(
                (
                    don_timestamp[-1] if don_timestamp.size > 0 else 0,
                    ka_timestamp[-1] if ka_timestamp.size > 0 else 0,
                )
            )
            safezone_keep = 0
            tja += f"COURSE:{i}\nLEVEL:{level}\n\n#START\n#SCROLL {scroll}\n"
            for time in range(length):
                if np.isin(time, don_timestamp) == True and safezone_keep <= 0:
                    tja += "1"
                    safezone_keep = safezone
                elif np.isin(time, ka_timestamp) == True and safezone_keep <= 0:
                    tja += "2"
                    safezone_keep = safezone
                else:
                    tja += "0"
                    safezone_keep -= 1
                if time % 100 == 99:
                    tja += ",\n"
            if length % 100 != 0:
                tja += "0" * (100 - (length % 100)) + ",\n"
            tja += "#END\n\n"
        except:
            pass

    return tja
