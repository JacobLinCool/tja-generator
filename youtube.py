import os
import gradio as gr
import yt_dlp
import tempfile
import hashlib


def youtube(url: str) -> str:
    if not url:
        raise gr.Error("Please input a YouTube URL")

    hash = hashlib.md5(url.encode()).hexdigest()
    tmp_file = os.path.join(tempfile.gettempdir(), f"{hash}")

    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": tmp_file,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise gr.Error(f"Failed to download YouTube audio from {url}")

    return tmp_file + ".mp3"
