#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install faster_whisper yt_dlp torch ipython nvidia-cublas-cu11 # faster_whisper
pip install transformers accelerate sentencepiece bitsandbytes # trurl
pip install tensorflow # misc
