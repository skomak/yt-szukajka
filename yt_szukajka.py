#!/usr/bin/env python
# coding: utf-8

import os, sys, glob, gc

print("Found text files:\n" + "\n".join([f for f in glob.glob("yt-*.txt")]))
print()
action = input("Provide a youtube link or text file you want to talk with: ")

import sys
import warnings
from faster_whisper import WhisperModel
from pathlib import Path
import yt_dlp
import subprocess
import torch
import shutil
import numpy as np
from IPython.display import display, Markdown, YouTubeVideo
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import wave
from transformers import logging
import warnings

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def transcribe(target):
    
    set_url = action
    set_lang = "auto" # auto, en, pl...
    set_model = "large-v2" # large-v2, medium, small...
    set_threads = 24 # default 4
    set_device = "cuda" # cpu, cuda
    set_compute = "int8" # 'float16', 'int8_float16', 'int8'
    set_torch_device = 'cuda:0'
    
    device = torch.device(set_torch_device)
    print('[i] Using device:', device, "for transcription", file=sys.stderr)
    
    model_size = set_model #'large-v2' #@param ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2']
    device_type = set_device #@param {type:"string"} ['cuda', 'cpu']
    compute_type = set_compute #@param {type:"string"} ['float16', 'int8_float16', 'int8']
    
    model = WhisperModel(model_size, device=device_type, compute_type=compute_type, cpu_threads=set_threads)
    
    Type = "Youtube video or playlist" #@param ['Youtube video or playlist', 'Google Drive']
    URL = target
    
    video_path_local_list = []
    
    if Type == "Youtube video or playlist":
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
    
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([URL])
            list_video_info = [ydl.extract_info(URL, download=False)]
            
        for video_info in list_video_info:
            video_path_local_list.append(Path(f"{video_info['id']}.wav"))
    else:
        raise(TypeError("Please select supported input type."))
    
    for video_path_local in video_path_local_list:
        if video_path_local.suffix == ".mp4":
            video_path_local = video_path_local.with_suffix(".wav")
            result  = subprocess.run(["ffmpeg", "-i", str(video_path_local.with_suffix(".mp4")), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(video_path_local)])
     
    def seconds_to_time_format(s):
        # Convert seconds to hours, minutes, seconds, and milliseconds
        hours = s // 3600
        s %= 3600
        minutes = s // 60
        s %= 60
        seconds = s // 1
        milliseconds = round((s % 1) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"
    
    start = time.time()
    f = wave.open(str(video_path_local),'r')
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    f.close()
    
    language = set_lang #"auto" #@param ["auto", "en", "zh", "ja", "fr", "de"] {allow-input: true}
    initial_prompt = "Please do not translate, only transcription be allowed"#.  Here are some English words you may need: Cindy. And Chinese words: \u7206\u7834" #@param {type:"string"}
    word_level_timestamps = False #@param {type:"boolean"}
    vad_filter = True #@param {type:"boolean"}
    vad_filter_min_silence_duration_ms = 50 #@param {type:"integer"}
    text_only = True # Output(Default is srt, txt if `text_only` be checked )
    
    
    segments, info = model.transcribe(str(video_path_local), beam_size=5,
                                      language=None if language == "auto" else language,
                                      initial_prompt=initial_prompt,
                                      word_timestamps=word_level_timestamps, 
                                      vad_filter=vad_filter,
                                      vad_parameters=dict(min_silence_duration_ms=vad_filter_min_silence_duration_ms))
    
    display(Markdown(f"Detected language '{info.language}' with probability {info.language_probability}"))
    
    ext_name = '.txt' if text_only else ".srt"
    transcript_file_name = "yt-" + video_path_local.stem + ext_name
    sentence_idx = 1
    with open(transcript_file_name, 'w') as f:
      for segment in segments:
        if word_level_timestamps:
          for word in segment.words:
            ts_start = seconds_to_time_format(word.start)
            ts_end = seconds_to_time_format(word.end)
            print(f"[{ts_start} --> {ts_end}] {word.word}")
            if not text_only:
              f.write(f"{sentence_idx}\n")
              f.write(f"{ts_start} --> {ts_end}\n")
              f.write(f"{word.word}\n\n")
            else:
              f.write(f"{word.word}")
            f.write("\n")
            sentence_idx = sentence_idx + 1
        else:
          ts_start = seconds_to_time_format(segment.start)
          ts_end = seconds_to_time_format(segment.end)
          print(f"[{ts_start} --> {ts_end}] {segment.text}")
          if not text_only:
            f.write(f"{sentence_idx}\n")
            f.write(f"{ts_start} --> {ts_end}\n")
            f.write(f"{segment.text.strip()}\n\n")
          else:
            f.write(f"{segment.text.strip()}\n")
          sentence_idx = sentence_idx + 1
    
    end = time.time()
    print("\n[i] Audio duration:", duration, "secs")
    print("[i] Transcription took:", end - start, "secs")
    
    del model
    gc.collect()
    
    return transcript_file_name

if action.startswith("http"):
    action = transcribe(action)

# do sth with transcription from transcript_file_name

start = time.time()

#tokenizer = AutoTokenizer.from_pretrained("Voicelab/trurl-2-7b")
#model = AutoModelForCausalLM.from_pretrained("Voicelab/trurl-2-7b", device_map=torch.device("cpu"))
#tokenizer = AutoTokenizer.from_pretrained("Voicelab/trurl-2-13b")
#model = AutoModelForCausalLM.from_pretrained("Voicelab/trurl-2-13b", device_map=torch.device("cpu"))
tokenizer = AutoTokenizer.from_pretrained("Voicelab/trurl-2-7b-8bit")
model = AutoModelForCausalLM.from_pretrained("Voicelab/trurl-2-7b-8bit", device_map='auto')

end = time.time()
print("[i] Loading model took", end - start, "secs")

prompt = """
<s>[INST] <<SYS>> Answer as truthfully using the context below and nothing more. If you don't know the answer, say \"don't know\". Answer concisely in polish language.\nContext:\nTEXT\n<</SYS>>

MYPROMPT[/INST]

"""

text = open(action, 'r').read()

def ans(prompt, instr):
    
    prompt = prompt.replace("TEXT", text)
    prompt = prompt.replace("MYPROMPT", instr)
    
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        # debug
        #print(tokenizer.decode(model.generate(tokenized_prompt.data["input_ids"], max_new_tokens=500, temperature=0)[0], skip_special_tokens=True))
        # prod
         answer = tokenizer.decode(model.generate(tokenized_prompt.data["input_ids"], max_new_tokens=500, temperature=0)[0], skip_special_tokens=True)
         print("Answer: ", answer.split("[/INST]")[1].lstrip())

print()
print()
print()
print("Type 'q' to quit.")
while True:
    print()
    instr = input("prompt> ")
    if instr == 'q':
        sys.exit()
    
    start = time.time()
    ans(prompt, instr)
    end = time.time()
    print("[i] Prompt took", end - start, "secs")














