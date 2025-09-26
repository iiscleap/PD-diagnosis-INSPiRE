print(f"Start of the file")
import librosa
import numpy as np
import os
from pydub import AudioSegment
from scipy.signal import find_peaks
import argparse


def list_of_strings(arg):
    return arg.split(',')

# --- Parameters ---
parser = argparse.ArgumentParser(description="Extract transitions with specified windows lengths and concatenate them into one audio")
parser.add_argument("--window_ms", type=int, required=True, help="Window size before and after the transition detected (in ms).")
parser.add_argument("--input_dir", type=str, required=True, help="Complete path to the full audio samples")
parser.add_argument("--output_dir", type=str, required=True, help="Complete path to the output directory")
parser.add_argument("--output_dir", type=str, required=True, help="Complete path to the output directory")
parser.add_argument('--folders', type=list_of_strings, required=True, help="Names of folders inside the input directory separated by comma(,)")

args = parser.parse_args()

FOLDER_NAME = args.folders
WINDOW_MS = int(args.window_ms)                              # Time before/after transition
USE_ONSET_DETECTION = True                   # Toggle between onset and RMS


print(f"Initializing parameters")

INPUT_DIR = args.input_dir
OUTPUT_DIR = os.path.join(args.output_dir, "16k_syllable_transitions_{WINDOW_MS}ms")


os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio(file_path, output_subdir):
    # Load audio
    y, sr = librosa.load(file_path)
    audio_db = AudioSegment.from_wav(file_path)

    # Transition detection
    if USE_ONSET_DETECTION:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        transition_times = librosa.frames_to_time(onset_frames, sr=sr)
    else:
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        peaks, _ = find_peaks(rms, prominence=0.01, distance=10)
        transition_times = times[peaks]

    print(f"[{os.path.basename(file_path)}] {len(transition_times)} transitions detected")

    # Output paths
    os.makedirs(output_subdir, exist_ok=True)
    combined_audio = AudioSegment.silent(duration=0)
    timestamps = []

    # Segment extraction
    for i, t in enumerate(transition_times):
        center_ms = int(t * 1000)
        start_ms = max(center_ms - WINDOW_MS, 0)
        end_ms = min(center_ms + WINDOW_MS, len(audio_db))
        segment = audio_db[start_ms:end_ms]

        # Save segment
        segment_path = os.path.join(output_subdir, f"segment_{i+1:03d}.wav")
        segment.export(segment_path, format="wav")

        # Add to combined audio
        combined_audio += segment  # + AudioSegment.silent(duration=100) if pause needed
        timestamps.append((start_ms / 1000, end_ms / 1000))

    # Save combined audio
    combined_path = os.path.join(output_subdir, "combined.wav")
    combined_audio.export(combined_path, format="wav")

    # Save a copy of combined audio into a central folder
    global_combined_dir = os.path.join(OUTPUT_DIR, "all_combined_wavs")
    os.makedirs(global_combined_dir, exist_ok=True)

    combined_basename = os.path.basename(output_subdir) + "_combined.wav"
    combined_global_path = os.path.join(global_combined_dir, combined_basename)
    combined_audio.export(combined_global_path, format="wav")

    # Save timestamps
    with open(os.path.join(output_subdir, "timestamps.txt"), "w") as f:
        for start, end in timestamps:
            f.write(f"{start:.2f}\t{end:.2f}\n")

    print(f"Processed and saved: {combined_path} and {global_combined_dir}")

if __name__ == "__main__":
    print(f"\nWindow length: {WINDOW_MS}\n")
    for folder in FOLDER_NAME:
        folder_path=os.path.join(INPUT_DIR, folder)
        print(f"folder_path:{folder_path}")
        for filename in os.listdir(folder_path):
            print(f"file_name:{filename}")
            if filename.lower().endswith("_clean.wav"):
                print(f"file_name ending with _clean.wav:{filename}")
                input_path = os.path.join(folder_path, filename)
                output_subdir = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0])
                process_audio(input_path, output_subdir)