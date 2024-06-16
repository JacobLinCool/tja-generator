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


def detection(don_inference, ka_inference, song, filepath):
    """detects notes disnotesiresultg don and ka"""

    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (
        peak_pick(
            x=don_inference,
            pre_max=1,
            post_max=2,
            pre_avg=4,
            post_avg=5,
            delta=0.05,
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
            delta=0.05,
            wait=3,
        )
        + 7
    )

    print(don_timestamp)
    print(ka_timestamp)

    song.don_timestamp = don_timestamp[
        np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])
    ]
    song.timestamp = song.don_timestamp * 512 / song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff="don")

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp[
        np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])
    ]
    song.timestamp = song.ka_timestamp * 512 / song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff="ka")

    song.save(filepath)


def create_tja(song, don_timestamp, ka_timestamp=None):
    tja = ""

    if ka_timestamp is None:
        timestamp = don_timestamp * 512 / song.samplerate
        tja += f"TITLE: untitled\nSUBTITLE: --\nBPM: 240\nWAVE:untitled.ogg\nOFFSET:0\n#START\n"
        i = 0
        time = 0
        while i < len(timestamp):
            if time / 100 >= timestamp[i]:
                tja += "1"
                i += 1
            else:
                tja += "0"
            if time % 100 == 99:
                tja += ",\n"
            time += 1
        tja += "\n#END"

    else:
        don_timestamp = np.rint(don_timestamp * 512 / song.samplerate * 100).astype(
            np.int32
        )
        ka_timestamp = np.rint(ka_timestamp * 512 / song.samplerate * 100).astype(
            np.int32
        )
        tja += f"TITLE: untitled\nSUBTITLE: --\nBPM: 240\nWAVE:untitled.ogg\nOFFSET:0\n#START\n"
        for time in range(np.max((don_timestamp[-1], ka_timestamp[-1]))):
            if np.isin(time, don_timestamp) == True:
                tja += "1"
            elif np.isin(time, ka_timestamp) == True:
                tja += "2"
            else:
                tja += "0"
            if time % 100 == 99:
                tja += ",\n"
        tja += "\n#END"

    return tja
