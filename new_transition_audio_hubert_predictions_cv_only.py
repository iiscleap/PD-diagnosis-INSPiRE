# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import re

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import ConfusionMatrixDisplay

from pdb import set_trace as bp
import logging

np.random.seed(42)


def list_of_strings(arg):
    return arg.split(',')

# --- Parameters ---
print(f"Initializing parameters")
parser = argparse.ArgumentParser(description="Predict Parkinson's disease status using baseline features and gridsearch")
parser.add_argument("--log_path", type=str, required=True, help="Path to save the log file at")
parser.add_argument("--window_ms", type=int, required=True, help="Window size before and after the transition detected (in ms).")
parser.add_argument("--embeddings_dir", type=str, required=True, help="Complete path to the full audio samples")
parser.add_argument("--output_dir", type=str, required=True, help="Complete path to the output directory")
parser.add_argument("--meta_dir", type=str, required=True, help="Complete path to the meta file (with labels columns 'park')")
parser.add_argument('--folders', type=list_of_strings, required=True, help="Names of folders inside the input directory separated by comma(,)")
parser.add_argument('--feat_sets', type=list_of_strings, required=True, help="Features to be extracted from the given audios separated by comma(,). Options avaialble: hubert_base, hubert_large_pretrained, hubert_large_finetuned, hubert_xlarge_pretrained, hubert_xlarge_finetuned")
parser.add_argument('--with_meta', type=bool, help="Do you want to concatenate the metadata with the features? Options: 0 or 1")
parser.add_argument('--device', type=bool, help="What device do you want to run the hubert model on? Options: cpu or cuda")
args = parser.parse_args()

log_file_path = args.log_path

logging.basicConfig(filename=log_file_path,
                    format='[%(filename)s:%(lineno)s - %(funcName)20s()][%(asctime)s] %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

WINDOW_MS = int(sys.argv[1])                              # Time before/after transition

FEATURE_NAMES = args.folders
FEATURE_MAPS = args.feat_sets
WITH_META=args.with_meta
NEW_ONSETS=0

device = torch.device(args.device)


### PCA
def pca_transform(data_pca):
    pca = PCA()
    data_pca = pca.fit_transform(data_pca.drop("park", axis=1))

    explained_var = pca.explained_variance_ratio_
    cum_sum = np.cumsum(explained_var)
    cum_sum_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(cum_sum))],
        "Cumulative Variance": cum_sum    
    })

    cum_sum_df.round(4).head(10)
    plt.figure()
    plt.bar(x=range(1, len(explained_var)+1), height=explained_var)
    plt.title('Variance of each Principal Components')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance')
    plt.grid(True)
    plt.savefig(f"PC_variance")



    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_pca.drop('park', axis=1))

    pca_plot = sns.scatterplot(data_pca)
    fig = pca_plot.get_figure()
    fig.savefig("pca_scatter.png")


### TSNE
def tsne_transform(data_tsne):
    tsne = TSNE()
    data_tsne_reduced = tsne.fit_transform(data_tsne.drop('park', axis=1))

    data_tsne_df = pd.DataFrame(data=data_tsne_reduced, columns=['tsne0', 'tsne1'])
    data_tsne_df['park'] = data_tsne['park'].values

    plt.figure()
    sns.scatterplot(data=data_tsne_df, x='tsne0', y='tsne1', hue='park')
    plt.title('Scatter plot for the data reduced using TSNE')
    plt.savefig(f"tsne_scatter")



### LDA
def lda_transform(data_lda):
    lda = LinearDiscriminantAnalysis()

    transformed_data_lda = lda.fit_transform(X=data_lda.drop('park', axis=1), y=data_lda['park'])
    data_lda_df = pd.DataFrame(transformed_data_lda, columns=lda.get_feature_names_out())
    data_lda_df['park'] = data_lda['park'].values

    plt.figure()

    plt.scatter(x=data_lda[data_lda['park']==0].index, y=data_lda[data_lda['park']==0]['lineardiscriminantanalysis0'],
                color='tab:orange',
                label='Healthy'
                )

    plt.scatter(x=data_lda[data_lda['park']==1].index, y=data_lda[data_lda['park']==1]['lineardiscriminantanalysis0'],
                color='tab:blue',
                label='PD diagnosed'
                )

    plt.legend(['Healthy', "PD Diagnosed"])
    plt.title('Scatter plot for the data reduced using LDA')
    plt.savefig(f"scatter_lda")
    
    return transformed_data_lda




def metrics_display(mdl, X_complete, y_complete, model_name, feat_name, feat_map, params, results_dict, layer_no, data_type='test'):
    #--------------------Cross validation with scores output (Confusion matrix not available)--------------------------

    scoring = ["balanced_accuracy", "accuracy", "precision", "recall", "f1_macro", "roc_auc"]
    logger.info(f"\nParameters being used for the grid search for {model_name}: {params}")
    grid = GridSearchCV(mdl, param_grid=params, scoring=scoring, cv=5, refit='balanced_accuracy')
    grid.fit(X_complete, y_complete)
    
    scores=grid.cv_results_
    best_result_index=grid.best_index_
    
    results_dict[feat_name]['Hubert Layer'].append(layer_no)
    results_dict[feat_name]['mean_balanced_acc'].append(scores['mean_test_balanced_accuracy'][best_result_index])
    results_dict[feat_name]['std_balanced_acc'].append(scores['std_test_balanced_accuracy'][best_result_index])
    results_dict[feat_name]['mean_f1_score'].append(scores['mean_test_f1_macro'][best_result_index])
    results_dict[feat_name]['std_f1_score'].append(scores['std_test_f1_macro'][best_result_index])
    results_dict[feat_name]['method'].append(feat_name)
    results_dict[feat_name]['transition_window_size'].append(WINDOW_MS)

    logger.info(f"\nAvailable score: {sorted(scores.keys())}\n")
    logger.info(f"The combination of hyperparameters giving the best accuracy is {grid.best_params_} with an accuracy of {grid.best_score_}\n")
    logger.info(f"The average metrics for prediction using {model_name} are")
    for metric_iter in sorted(scores.keys()):
        if metric_iter.startswith(('mean_','std_')) and not metric_iter.endswith(('fit_time', 'score_time', 'estimator')):
            logger.info(f"{metric_iter}: {scores[metric_iter][best_result_index]}")
    logger.info("\n\n\n")

    return results_dict
    
    #------------------------------------------------------------------------------------------------------------------


def pool_embeddings(embed, layer_no, max_batch = None):
    use_embed = embed[layer_no]
    
    return torch.mean(use_embed, axis=0)

class HubertEmbeddingDataset(Dataset):
    def __init__(self, aud_names, folder_name, meta, embed_dir, layer_no, feat_map, max_frames_batch=0):
        self.aud_names = aud_names
        self.folder_name = folder_name
        self.meta = meta
        self.embed_dir = embed_dir
        self.max_frames_batch_count = max_frames_batch
        self.layer_no = layer_no
        self.feat_map = feat_map

    def __len__(self):
        return len(self.aud_names)

    def __getitem__(self, idx):
        aud_name = self.aud_names[idx]

        if self.feat_map != "hubert_base":
            embed_path = os.path.join(self.embed_dir, f"{self.feat_map}_16k_full_features_{aud_name}_{self.folder_name}_features_{self.feat_map}.csv")
        elif self.feat_map == "hubert_base":

            embed_path = os.path.join(self.embed_dir, f"hubert_16k_transition_{WINDOW_MS}ms_{aud_name}_{self.folder_name}_features_hubert.csv")
            
        try:
            with open(embed_path, "rb") as fp:
                embed = torch.load(fp, map_location=torch.device('cpu'))

            pooled_embed = pool_embeddings(embed, self.layer_no, max_batch = self.max_frames_batch_count)
                    
            pooled_embed = pooled_embed.to(device)
        
        except:
            pooled_embed = -1

        if WITH_META==0:
            label = torch.tensor(self.meta.loc[aud_name], dtype=torch.float32)
        elif (WITH_META==1) and (torch.all(pooled_embed!=torch.tensor([-1]))):
            label = torch.tensor(self.meta.loc[aud_name, 'park'], dtype=torch.float32)
            clinical_info = self.meta.drop(['park'], axis=1)
            pooled_embed=torch.cat(pooled_embed, torch.tensor(clinical_info[aud_name].to_numpy()))

        return pooled_embed, label


class ClassifierHead(nn.Module):
    def __init__(self, inp_shape):
        super(ClassifierHead, self).__init__()
        self.inp_shape=inp_shape
        self.hidden_shape=4*self.inp_shape

        self.ff1 = nn.Linear(in_features=self.inp_shape, out_features=self.hidden_shape)
        self.ff2 = nn.Linear(in_features=self.hidden_shape, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, embeds):
        logits = self.relu(self.ff1(embeds))
        return self.ff2(logits)


def filter_vid_names(emb_path, meta_series, feat_map):
    file_list = os.listdir(emb_path)

    uuid_counts = defaultdict(int)

    for filename in file_list:
        if NEW_ONSETS:
            if re.match(r"aubio_.*_.*_features_aubio_.*_hubert\.csv", filename):
                parts = filename.split('_')
                try:
                    if feat_map != "hubert_base":
                        uuid = parts[6]
                        uuid_counts[uuid] += 1
                    elif feat_map == "hubert_base":
                        if not NEW_ONSETS:
                            uuid = parts[4]
                            uuid_counts[uuid] += 1
                        elif NEW_ONSETS:
                            uuid = parts[1]
                            uuid_counts[uuid] += 1
                except IndexError:
                    continue
        elif not NEW_ONSETS:
            parts = filename.split('_')
            try:
                if feat_map != "hubert_base":
                    uuid = parts[6]
                    uuid_counts[uuid] += 1
                elif feat_map == "hubert_base":
                    if not NEW_ONSETS:
                        uuid = parts[4]
                        uuid_counts[uuid] += 1
                    elif NEW_ONSETS:
                        uuid = parts[1]
                        uuid_counts[uuid] += 1
            except IndexError:
                continue

    valid_uuids = [uuid for uuid, count in uuid_counts.items() if count == 5]
    return meta_series[meta_series.index.isin(valid_uuids)]


if __name__ == "__main__":
    logger.info(f"Main function started with numpy version is {np.__version__} and device:{device}")
    corr_threshold=0.1

    for FEATURE_MAP in FEATURE_MAPS:
        logger.info(f"\nHubert embeddings from {FEATURE_MAP}\n")
        results_dict = {key: {items:[] for items in ['Hubert Layer', 'mean_balanced_acc', 'std_balanced_acc', 'mean_f1_score', 'std_f1_score', 'method', 'transition_window_size']} for key in FEATURE_NAMES}
        
        if FEATURE_MAP in ['hubert_large_pretrained', 'hubert_large_finetuned']:
            max_layers = 25
        elif FEATURE_MAP in ['hubert_xlarge_pretrained', 'hubert_xlarge_finetuned']:
            max_layers = 49
        else:
            max_layers = 13

        for LAYER_NO in range(0,max_layers):
            logger.info(f"\Predictions with hubert embeddings of 16kHz sampled audios with transition videos with window size {WINDOW_MS} with {FEATURE_MAP} layer {LAYER_NO} with CV and GRIDSEARCHCV\n")

            for FEATURE_NAME in FEATURE_NAMES:
                if FEATURE_MAP != "hubert_base":
                    hubert_emb_path = os.path.join(args.embeddings_dir, f"{FEATURE_MAP}_16k_full_features")
                elif FEATURE_MAP == "hubert_base":
                    if not NEW_ONSETS:
                        hubert_emb_path = os.path.join(args.embeddings_dir, f"hubert_16k_transition_{WINDOW_MS}ms_features")
                    elif NEW_ONSETS:
                        hubert_emb_path = os.path.join(args.embeddings_dir, f"\hubert_16k_onset_methods_{WINDOW_MS}")

                meta_path = os.path.join(args.meta_dir)

                logger.info(f"\nExperiment with {FEATURE_NAME} and pretrained {FEATURE_MAP} embeddings from layer {LAYER_NO}\n")


                meta = pd.read_csv(meta_path, index_col='id')

                meta_cols = list(meta.columns)

                class_dist = meta['park'].value_counts()
                
                fig, ax = plt.subplots()
                class_dist_bar = plt.bar(x=['yes', 'no'], height=class_dist)
                ax.bar_label(class_dist_bar, label_type='center')

                filtered_meta = filter_vid_names(hubert_emb_path, meta, FEATURE_MAP)
                
                meta_yes = filtered_meta[filtered_meta['park'] == 'yes']
                meta_no = filtered_meta[filtered_meta['park'] == 'no']

                meta_yes=meta_yes.sort_values(by=["hoehn", "updrs"], ascending=False)
                meta_yes_balanced = meta_yes.head(len(meta_no))

                meta_balanced = pd.concat([meta_yes_balanced, meta_no])
                meta_balanced = meta_balanced.sample(frac=1, random_state=42)
                class_dist = meta_balanced['park'].value_counts()


                meta_cols.remove('park')
                # ------------------------------------------------------------------------------------------------------------------ #
                if WITH_META==0:
                    meta_balanced.drop(meta_cols, axis=1, inplace=True)
                # ------------------------------------------------------------------------------------------------------------------ #

                # ------------------------------------------------------------------------------------------------------------------ #
                if WITH_META==1:
                    meta_balanced.drop(["hoehn", "updrs"], axis=1, inplace=True)
                    meta_balanced.drop(["moca", "levo", "duration", "date"], axis=1, inplace=True)
                    meta_balanced['loc'] = meta_balanced['loc'].map({'Bangalore': 'Bengaluru', 'Bengaluru, Karnataka':'Bengaluru'})
                    meta_balanced.drop(['old_info', 'od_info'], axis=1, inplace=True)
                    meta_balanced = meta_balanced.drop(['name', 'date:time'], axis=1)
                # ------------------------------------------------------------------------------------------------------------------ #

                meta_balanced['park'] = meta_balanced['park'].map({'yes':1, 'no':0})

                # ------------------------------------------------------------------------------------------------------------------ #
                if WITH_META==1:
                    categorical_cols = list(meta_balanced.select_dtypes(include=[object]).columns)

                    label_enc = {}
                    for col in categorical_cols:
                        label_enc[col] = LabelEncoder()
                        meta_balanced[col] = label_enc[col].fit_transform(meta_balanced[col])


                    meta_balanced = meta_balanced.fillna(-1)
                # ------------------------------------------------------------------------------------------------------------------ #

                train_meta, val_meta = train_test_split(
                    meta_balanced,
                    test_size=0.2,
                    stratify=meta_balanced['park'],
                    random_state=42
                )


                fig, ax = plt.subplots()
                class_dist_bar = plt.bar(x=['yes', 'no'], height=class_dist)
                ax.bar_label(class_dist_bar, label_type='center')

                time_format = "%H:%M:%S.%f"

                train_dataset = HubertEmbeddingDataset(
                    aud_names=list(train_meta.index),
                    meta=train_meta,
                    folder_name=FEATURE_NAME,
                    embed_dir=hubert_emb_path,
                    layer_no=LAYER_NO,
                    feat_map=FEATURE_MAP
                )

                val_dataset = HubertEmbeddingDataset(
                    aud_names=list(val_meta.index),
                    meta=val_meta,
                    folder_name=FEATURE_NAME,
                    embed_dir=hubert_emb_path,
                    layer_no=LAYER_NO,
                    feat_map=FEATURE_MAP
                )

                complete_dataset = HubertEmbeddingDataset(
                    aud_names=list(meta_balanced.index),
                    meta=meta_balanced,
                    folder_name=FEATURE_NAME,
                    embed_dir=hubert_emb_path,
                    layer_no=LAYER_NO,
                    feat_map=FEATURE_MAP
                )

                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                complete_loader = DataLoader(complete_dataset, batch_size=1, shuffle=False)


                logger.info(f"Shape of the embeddings: {next(iter(train_loader))[0].shape}\n")
                
                embed_df=pd.DataFrame([])
                for embeds, labels in complete_loader:
                    embeds = pd.DataFrame(embeds.detach().numpy())
                    embeds['park']=labels
                    embed_df = pd.concat([embed_df, embeds], ignore_index=True)
                
                X, y = embed_df.drop('park', axis=1), embed_df['park']

                ### Classifying the data using RandomForest
                rf_model = RandomForestClassifier(class_weight='balanced')
                rf_params = {'n_estimators':[50,150,250,350,450],
                            'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
                            }
                

                results_dict = metrics_display(rf_model, X, y, "Random Forest Classifier", FEATURE_NAME, FEATURE_MAP, params=rf_params, results_dict=results_dict, layer_no=LAYER_NO)

            
        print(FEATURE_MAP, repr(results_dict))
        logger.info(f"{FEATURE_MAP} with new onsets={NEW_ONSETS}, transition window {WINDOW_MS}ms: {repr(results_dict)}")