# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

RUN useradd -m -u 1000 user

WORKDIR /app

RUN apt update && apt install -y curl libsndfile1 ffmpeg

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

RUN mkdir -p /app/models && \
    curl -L https://huggingface.co/JacobLinCool/odcnn-320k-100/resolve/main/don_model.pth -o /app/models/don_model.pth && \
    curl -L https://huggingface.co/JacobLinCool/odcnn-320k-100/resolve/main/ka_model.pth -o /app/models/ka_model.pth && \
    chown -R user /app/models

CMD ["python", "app.py"]
