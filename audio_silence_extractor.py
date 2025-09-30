import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse
from datetime import datetime as dt
import datetime

from sklearn import tree

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import re

import os
import subprocess
from scipy.io import wavfile

np.random.seed(42)

def list_of_strings(arg):
    return arg.split(',')

# --- Parameters ---
print(f"Initializing parameters")
parser = argparse.ArgumentParser(description="Extract features from given audios")
parser.add_argument("--video_data_dir", type=str, required=True, help="Complete path to the full videos from which the audios have to be extracted and silences (initial and trailing) to be removed")
parser.add_argument("--audio_data_dir", type=str, required=True, help="Complete path to the store the audio files")
parser.add_argument('--audio_silence_removed_dir', type=str, required=True, help="Complete path to the store the audio files")
parser.add_argument('--sox_path', type=str, required=True, help="Root path for sox")
parser.add_argument("--ffmpeg_path", type=str, required=True, help="Root path for ffmpeg")

args = parser.parse_args()

if __name__ == "__main__":
    print(f"Start of the script")
    video_data_dir = args.video_data_dir
    audio_data_dir = args.audio_data_dir
    silence_removed_dir = args.silence_removed_dir
    sox_path = args.sox_path
    ffmpeg_path = args.ffmpeg_path

    for folder in os.listdir(video_data_dir):
        video_files_path = os.path.join(video_data_dir, folder)
        audio_files_path = os.path.join(audio_data_dir, folder)

        if not os.path.isdir(video_files_path) or folder.endswith('.csv'):
            continue

        os.makedirs(audio_files_path, exist_ok=True)

        for mp4_file in os.listdir(video_files_path):
            if not mp4_file.endswith(".mp4"):
                continue

            file_path = os.path.join(video_files_path, mp4_file)
            base_name = os.path.splitext(mp4_file)[0]
            raw_audio_path = os.path.join(audio_files_path, base_name + "_raw.wav")
            clean_audio_path = os.path.join(audio_files_path, base_name + "_clean.wav")

            try:
                print(f"[FFMPEG] Converting {file_path} to {raw_audio_path}")
                subprocess.run([
                    ffmpeg_path, "-i", file_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "1", raw_audio_path
                ], check=True)

                print(f"[SOX] Removing silence from {raw_audio_path}")
                subprocess.run([
                    sox_path, raw_audio_path, clean_audio_path,
                    "silence", "1", "0.1", "1%", "reverse",
                    "silence", "1", "0.1", "1%", "reverse"
                ], check=True)

                sr, _ = wavfile.read(clean_audio_path)
                print(f"[INFO] {clean_audio_path} sampling rate: {sr} Hz")
                # print(f"[INFO] Sampling rate: {sr}")

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Conversion failed for {file_path}: {e}")
            except Exception as ex:
                print(f"[ERROR] Unexpected error: {ex}")
