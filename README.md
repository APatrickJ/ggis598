# ggis598
Graduate Capstone Project

## Setup

```shell
pyenv virtualenv 3.11.12 ggis598
pyenv activate ggis598
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

## Build Docker image

From an Apple Silicon Mac:

```shell
docker buildx build --platform=linux/amd64 -t lfmc-train-$(date "+%Y%m%d-%H%M%S") .
```
