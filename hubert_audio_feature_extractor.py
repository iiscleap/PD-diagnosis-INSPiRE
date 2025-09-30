import librosa
import torch
import sys

import opensmile
import os
import argparse

import numpy as np
import pandas as pd
from collections import defaultdict
from pdb import set_trace as bp

from transformers import AutoProcessor, HubertModel
import torch

from pathlib import Path  

def list_of_strings(arg):
    return arg.split(',')

# --- Parameters ---
print(f"Initializing parameters")
parser = argparse.ArgumentParser(description="Extract hubert-base features from given audios")
parser.add_argument("--input_dir", type=str, required=True, help="Complete path to the audio samples")
parser.add_argument("--output_dir", type=str, required=True, help="Complete path to the output directory")
parser.add_argument("--window_ms", type=int, required=True, help="Window size before and after the transition detected (in ms).")
parser.add_argument('--folders', type=list_of_strings, required=True, help="Names of folders inside the input directory separated by comma(,)")
parser.add_argument('--feat_sets', type=list_of_strings, required=True, help="Features to be extracted from the given audios separated by comma(,). Options avaialble: egemaps, gemaps, compare, mel-spectrogram, mfcc")
parser.add_argument("--use_transition_audios", type=int, required=True, help="Do you want to use transition audios or full audios? Enter 0 for full audios and 1 for transtion audios")
parser.add_argument('--device', type=str, help="What device do you want to use to run the hubert model? Options: cpu or cuda")

args = parser.parse_args()

FOLDER_NAME = args.folders
FEAT_MAPS = args.feat_sets
WINDOW_MS = args.window_ms

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


WINDOW_MS = int(sys.argv[1])                              # Time before/after transition
FEAT_MAPS = ['hubert']
TRANSITION_AUDIOS=args.use_transition_audios

device = torch.device(args.device)

INPUT_DIR = os.path.join(args.input_dir, f"all_combined_wavs")

OUTPUT_DIR = os.path.join(args.output_dir)


os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_feats(feat_extractor, path, feat_map):
    return feat_extractor.process_file(path)
    
def extract_feats_hubert(feat_extractor, path, feat_map, processor=None):
    aud=np.array([])
    aud, sr = librosa.load(path, sr=16000)
    if len(aud)>0:
        return feat_extractor(torch.from_numpy(aud).unsqueeze(0).to(device), output_hidden_states=True)
    else:
        return {'hidden_states':[]}

def get_extractor(feat_map):
    if feat_map=='hubert':
        print(f"MAKE SURE TO USE 16KHZ AUDIOS AS HUBERT IS PRETRAINED ON 16KHZ AUDIOS!")
        ext = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
        return ext, None
    
    return ext


if __name__ == "__main__":
    for feat_map in FEAT_MAPS:
        print(f"Started etxraction script\n")

        if feat_map=='hubert':
            ext_obj, processor = get_extractor(feat_map)

        feats_df={"aa":defaultdict(),
                    "count":defaultdict(),
                    "ee":defaultdict(),
                    "PaTaKa":defaultdict(),
                    "u":defaultdict(),
                    }
        
        if TRANSITION_AUDIOS==0:
            for folder in FOLDER_NAME:
                audios_path = os.path.join(INPUT_DIR, folder)
                for filename in os.listdir(audios_path):
                    if filename.endswith("_clean.wav"):
                        print(f"Extracting features for {filename}")
                        folder = os.path.split(filename)[-1].split('_')[1]
                        feats_file_name=os.path.split(filename)[-1].split('_')[0]

                        input_path = os.path.join(audios_path, filename)

                        if feat_map=='hubert':
                            hidden_states = extract_feats_hubert(ext_obj, input_path, feat_map, processor=processor)
                            if hidden_states['hidden_states']:
                                feats = torch.concatenate(hidden_states['hidden_states'])
                                print(f"feats.shape:{feats.shape}")
                            
                        if hidden_states['hidden_states']:
                            filepath = Path(OUTPUT_DIR+"_"+feats_file_name+"_"+folder+f"_features_{feat_map}.csv")
                            torch.save(feats, filepath)
        
        elif TRANSITION_AUDIOS==1:
            audios_path = os.path.join(INPUT_DIR)
            for filename in os.listdir(audios_path):
                folder=filename.split('_')[-2]
                if filename.endswith("_clean_combined.wav"):
                    print(f"Extracting features for {filename}")
                    folder = os.path.split(filename)[-1].split('_')[1]
                    feats_file_name=os.path.split(filename)[-1].split('_')[0]

                    input_path = os.path.join(audios_path, filename)

                    if feat_map=='hubert':
                        hidden_states = extract_feats_hubert(ext_obj, input_path, feat_map, processor=processor)
                        if hidden_states['hidden_states']:
                            feats = torch.concatenate(hidden_states['hidden_states'])
                            print(f"feats.shape:{feats.shape}")
                        
                    if hidden_states['hidden_states']:
                        filepath = Path(OUTPUT_DIR+"_"+feats_file_name+"_"+folder+f"_features_{feat_map}.csv")
                        torch.save(feats, filepath)