import os
import argparse
import numpy as np
import pandas as pd
from pdb import set_trace as bp

import opensmile
from transformers import AutoProcessor, AutoModel
import torch
import librosa

from pathlib import Path  
import sys


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
parser.add_argument("--save_csv", type=int, required=True, help="Do you want to save the features as a csv file? Options: 0 or 1")
parser.add_argument("--use_transition_audios", type=int, required=True, help="Do you want to use transition audios or full audios? Enter 0 for full audios and 1 for transtion audios")
parser.add_argument('--device', type=bool, help="What device do you want to run the hubert model on? Options: cpu or cuda")

args = parser.parse_args()



FOLDER_NAME = args.folders
WINDOW_MS = args.window_ms                              # Time before/after transition
FEAT_MAPS = args.feat_sets
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

SAVE_CSV=0
SAVE_PICKLE=1


INPUT_DIR = os.path.join(args.input_dir, f"all_combined_wavs")

OUTPUT_DIR = os.path.join(args.output_dir)


os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_feats(feat_extractor, path, feat_map):
    if feat_map in ['compare', 'gemaps', 'egemaps']:
        return feat_extractor.process_file(path)
    elif feat_map=='mel-spectrogram':
        aud, sr=librosa.load(path, sr=16000)
        return feat_extractor(y=aud, sr=sr, n_mels=128)
    elif feat_map=='mfcc':
        aud, sr=librosa.load(path, sr=16000)
        return feat_extractor(y=aud, sr=sr, n_mfcc=39)

def get_extractor(feat_map):
    if feat_map=='compare':
        ext = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )


    if feat_map=='gemaps':
        ext = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,
        )


    elif feat_map=='egemaps':
        ext = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    elif feat_map=='mel-spectrogram':
        ext=librosa.feature.melspectrogram

    elif feat_map=='mfcc':
        ext=librosa.feature.mfcc
    
    return ext


if __name__ == "__main__":
    print(f"\nWindow length: {WINDOW_MS}\n\n")
    for FEAT_MAP in FEAT_MAPS:
        print(f"Started etxraction script for {FEAT_MAP}\n")

        ext_obj = get_extractor(FEAT_MAP)

        feats_df={"aa":pd.DataFrame([]),
                    "count":pd.DataFrame([]),
                    "ee":pd.DataFrame([]),
                    "PaTaKa":pd.DataFrame([]),
                    "u":pd.DataFrame([]),
                    }
        


        for filename in os.listdir(INPUT_DIR):
            folder=filename.split('_')[-2]
            
            if filename.endswith("_clean_combined.wav"):
                input_path = os.path.join(INPUT_DIR, filename)
                print(f"Extracting transition features for {filename}")
                folder = os.path.split(filename)[-1].split('_')[1]

                if FEAT_MAP in ['egemaps', 'gemaps', 'compare']:
                    feats = extract_feats(ext_obj, input_path, FEAT_MAP).reset_index().drop(["start", "end"], axis=1)

                    feats['file']=os.path.split(filename)[-1].split('_')[0]
                
                elif FEAT_MAP in 'mel-spectrogram':
                    extracted_feats = extract_feats(ext_obj, input_path, FEAT_MAP)
                    file = os.path.split(filename)[-1].split('_')[0]
                    
                    feats = pd.DataFrame.from_dict({
                        'file': file, 
                        'feats': [extracted_feats]
                    })

                    
                    feats.reset_index()
                
                elif FEAT_MAP in 'mfcc':
                    extracted_feats = extract_feats(ext_obj, input_path, FEAT_MAP)
                    file = os.path.split(filename)[-1].split('_')[0]
                    
                    feats = pd.DataFrame.from_dict({
                        'file': file, 
                        'feats': [extracted_feats]
                    })

                    
                    feats.reset_index()
                
                feats_df[folder] = pd.concat([feats_df[folder], feats], ignore_index=True)
            

        print(f"feats_df:{feats_df}")
            
        for i,_ in feats_df.items():
            if SAVE_CSV:
                filepath = Path(OUTPUT_DIR+"_"+i+f"_features_{FEAT_MAP}.csv")
                feats_df[i].to_csv(filepath)
            if SAVE_PICKLE:
                filepath = Path(OUTPUT_DIR+"_"+i+f"_features_{FEAT_MAP}.pkl")
                feats_df[i].to_pickle(filepath)