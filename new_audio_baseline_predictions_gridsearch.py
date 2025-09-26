# -*- coding: utf-8 -*-

import librosa

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import argparse

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

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
parser.add_argument("--input_dir", type=str, required=True, help="Complete path to the features")
parser.add_argument("--meta_dir", type=str, required=True, help="Complete path to the meta file (with labels columns 'park')")
parser.add_argument('--folders', type=list_of_strings, required=True, help="Names of folders inside the input directory separated by comma(,)")
parser.add_argument('--feat_sets', type=list_of_strings, required=True, help="Features to be extracted from the given audios separated by comma(,). Options avaialble: egemaps, gemaps, compare, mel-spectrogram, mfcc")
parser.add_argument('--with_meta', type=bool, help="Do you want to concatenate the metadata with the features? Options: 0 or 1")
args = parser.parse_args()

log_file_path = args.log_path

logging.basicConfig(filename=log_file_path,
                    format='[%(filename)s:%(lineno)s - %(funcName)20s()][%(asctime)s] %(message)s',
                    filemode='a')
logger = logging.getLogger()
logger.setLevel(logging.INFO)



FEATURE_NAMES = args.folders
FEATURE_MAPS = args.feat_sets
WITH_META=0


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
    data_pca = pca.fit_transform(data.drop('park', axis=1))

    pca_plot = sns.scatterplot(data_pca)
    fig = pca_plot.get_figure()
    fig.savefig("pca_scatter.png")


### TSNE
def tsne_transform(data_tsne):
    tsne = TSNE()
    data_tsne_reduced = tsne.fit_transform(data_tsne.drop('park', axis=1))

    data_tsne_df = pd.DataFrame(data=data_tsne_reduced, columns=['tsne0', 'tsne1'])
    data_tsne_df['park'] = data['park'].values

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




def metrics_display(mdl, X_complete, y_complete, model_name, feat_name, feat_map, params, results_dict, data_type='test'):

    #--------------------Cross validation with scores output (Confusion matrix not available)--------------------------

    scoring = ["balanced_accuracy", "accuracy", "precision", "recall", "f1_macro", "roc_auc"]
    
    logger.info(f"\nParameters being used for the grid search for {model_name}: {params}")

    data_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(mdl, param_grid=params, scoring=scoring, cv=data_splitter, refit='balanced_accuracy')
    grid.fit(X_complete, y_complete)
    
    scores=grid.cv_results_
    best_result_index=grid.best_index_


    results_dict[feat_name]['Feature'].append(feat_map)
    results_dict[feat_name]['mean_balanced_acc'].append(scores['mean_test_balanced_accuracy'][best_result_index])
    results_dict[feat_name]['std_balanced_acc'].append(scores['std_test_balanced_accuracy'][best_result_index])
    results_dict[feat_name]['mean_f1_score'].append(scores['mean_test_f1_macro'][best_result_index])
    results_dict[feat_name]['std_f1_score'].append(scores['std_test_f1_macro'][best_result_index])
    results_dict[feat_name]['method'].append(feat_name)
    results_dict[feat_name]['transition_window_size'].append("full_audio")

    
    logger.info(f"\nAvailable score: {sorted(scores.keys())}")
    logger.info(f"\nThe combination of hyperparameters giving the best accuracy is {grid.best_params_} with an accuracy of {grid.best_score_}")
    logger.info(f"\nThe average metrics for prediction using {model_name} are")
    for metric_iter in sorted(scores.keys()):
        if metric_iter.startswith(('mean_','std_')) and not metric_iter.endswith(('fit_time', 'score_time', 'estimator')):
            logger.info(f"{metric_iter}: {scores[metric_iter][best_result_index]}")
    logger.info("\n\n\n")

    # --------------------------------------- Collect probabilities for fusion ---------------------------------------
    best_params = grid.best_params_
    mdl.set_params(**best_params)

    probs = cross_val_predict(
                mdl,                # fresh base estimator with best params
                X_complete, y_complete,
                cv=data_splitter,
                method="predict_proba"
            )[:, 1]

    if 'fusion_probs' not in results_dict[feat_name]:
        results_dict[feat_name]['fusion_probs'] = {}
    results_dict[feat_name]['fusion_probs'][feat_map] = probs
    
    return results_dict
    #------------------------------------------------------------------------------------------------------------------


def get_X_feats(raw_inp_feats, labels):
    feats_list = []
    labels_list = []

    empty_inds = []
    for arr_name, arr, lbl in zip(raw_inp_feats.index, raw_inp_feats.values, labels):
        inp_feats = np.array(arr[0])

        if np.all(inp_feats == 0):
            empty_inds.append(arr_name)
            continue

        mel_mean = np.mean(inp_feats, axis=1)
        mel_std = np.std(inp_feats, axis=1)

        if inp_feats.shape[1] >= 3:
            if inp_feats.shape[1] >=9:
                width = 9
            else:
                width = inp_feats.shape[1] if inp_feats.shape[1] %2==1 else inp_feats.shape[1] -1

            delta = librosa.feature.delta(inp_feats, order=1, width=width)
            delta2 = librosa.feature.delta(inp_feats, order=2, width=width)

            delta_mean = np.mean(delta, axis=1)
            delta_std = np.std(delta, axis=1)

            delta2_mean = np.mean(delta2, axis=1)
            delta2_std = np.std(delta2, axis=1)

            feats_list.append(np.concatenate([
                mel_mean, mel_std,
                delta_mean, delta_std,
                delta2_mean, delta2_std
            ]))

            labels_list.append(lbl)
        else:
            continue

    print(f"Empty feats: {empty_inds}")
    return np.array(feats_list), np.array(labels_list)


if __name__ == "__main__":
    corr_threshold=0.1

    results_dict = {key: {items:[] for items in ['Feature', 'mean_balanced_acc', 'std_balanced_acc', 'mean_f1_score', 'std_f1_score', 'method', 'transition_window_size']} for key in FEATURE_NAMES}
    for FEATURE_NAME in FEATURE_NAMES:
        for FEATURE_MAP in FEATURE_MAPS: 
            if FEATURE_MAP in ['egemaps', 'gemaps', 'compare']:
                data_path = os.path.join(args.input_dir, f"\\16kcomplete_opensmile_features_{FEATURE_NAME}_features_{FEATURE_MAP}.csv")
                data = pd.read_csv(data_path).drop('Unnamed: 0', axis=1)
            elif FEATURE_MAP in ['mel-spectrogram', 'mfcc']:
                data_path = os.path.join(args.input_dir, f"\\16kcomplete_opensmile_features_{FEATURE_NAME}_features_{FEATURE_MAP}.pkl")
                data = pd.read_pickle(data_path)

            meta_path = os.path.join(args.meta_dir)


            logger.info(f"\nExperiment with {FEATURE_NAME} and {FEATURE_MAP}\n")

            data['file'] = data['file'].str.split('_', expand=True)[0]
            data.rename(columns={'file':'id'}, inplace=True)
            data = data.set_index('id')
            logger.info(f"Number of available in {FEATURE_MAP} is {len(data.columns)}")

            meta = pd.read_csv(meta_path, index_col='id')
            data = data.join(meta, on='id', how='left')
            
            meta_cols = list(meta.columns)
            
            # ------------------------------------------------------------------------------------------------------------------ #
            if WITH_META==0:
                data.drop(meta_cols, axis=1, inplace=True)
            # ------------------------------------------------------------------------------------------------------------------ #   

            class_dist = data['park'].value_counts()
            
            fig, ax = plt.subplots()
            class_dist_bar = plt.bar(x=['yes', 'no'], height=class_dist)
            ax.bar_label(class_dist_bar, label_type='center')
            plt.savefig(f"class_dist_imbalanced")

            data.sort_values(by=["hoehn", "updrs"], ascending=False, inplace=True)
            data_yes = data[data['park'] == 'yes']
            data_no = data[data['park'] == 'no']

            data_yes=data_yes.sort_values(by=["hoehn", "updrs"], ascending=False)
            data_yes_balanced = data_yes.head(len(data_no))

            data_balanced = pd.concat([data_yes_balanced, data_no])
            data = data_balanced.sample(frac=1, random_state=42)
            class_dist = data['park'].value_counts()


            meta_cols.remove('park')
            # ------------------------------------------------------------------------------------------------------------------ #
            if WITH_META==0:
                data.drop(["hoehn", "updrs"], axis=1, inplace=True)
            # ------------------------------------------------------------------------------------------------------------------ #

            # ------------------------------------------------------------------------------------------------------------------ #
            if WITH_META==1:
                data.drop(["hoehn", "updrs"], axis=1, inplace=True)
                data.drop(["moca", "levo", "duration", "date"], axis=1, inplace=True)
                data['loc'] = data['loc'].map({'Bangalore': 'Bengaluru', 'Bengaluru, Karnataka':'Bengaluru'})
                data.drop(['old_info', 'od_info'], axis=1, inplace=True)
                data = data.drop(['name', 'date:time'], axis=1)
            # ------------------------------------------------------------------------------------------------------------------ #

            data['park'] = data['park'].map({'yes':1, 'no':0})

            # ------------------------------------------------------------------------------------------------------------------ #
            if WITH_META==1:
                categorical_cols = list(data.select_dtypes(include=[object]).columns)

                label_enc = {}
                for col in categorical_cols:
                    label_enc[col] = LabelEncoder()
                    data[col] = label_enc[col].fit_transform(data[col])


                data = data.fillna(-1)
            # ------------------------------------------------------------------------------------------------------------------ #

            X, y = data.drop('park', axis=1), data['park']

            if FEATURE_MAP in ['mel-spectrogram', 'mfcc']:
                X, y=get_X_feats(X, y)

            ### Classifying the data using RandomForest
            rf_model = RandomForestClassifier(class_weight='balanced')
            
            rf_params = {'n_estimators':[50,150,250,350,450],
                      'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
                      }

            results_dict = metrics_display(rf_model, X, y, "Random Forest Classifier", FEATURE_NAME, FEATURE_MAP, params=rf_params, results_dict=results_dict)
            rf_model.fit(X,y)
    




    # --- Fusion across FEATURE_NAMES (single FEATURE_MAP, no map averaging) ---
    for FEATURE_MAP in FEATURE_MAPS:
        task_probs_list = []
        for FEATURE_NAME in FEATURE_NAMES:
            if 'fusion_probs' in results_dict[FEATURE_NAME] and FEATURE_MAP in results_dict[FEATURE_NAME]['fusion_probs']:
                task_probs_list.append(results_dict[FEATURE_NAME]['fusion_probs'][FEATURE_MAP])
        
        if task_probs_list:
            all_tasks_probs = np.column_stack(task_probs_list)
            final_probs = all_tasks_probs.mean(axis=1)
            final_pred = (final_probs > 0.5).astype(int)

            results_dict[f'final_fusion_{FEATURE_MAP}'] = {
                'probs': final_probs,
                'balanced_acc': balanced_accuracy_score(y, final_pred),
                'f1_score': f1_score(y, final_pred),
                'accuracy': accuracy_score(y, final_pred)
            }

            logger.info(f"Fusion across FEATURE_NAMES using {FEATURE_MAP} only: "
                        f"BA={results_dict[f'final_fusion_{FEATURE_MAP}']['balanced_acc']:.3f}, "
                        f"F1={results_dict[f'final_fusion_{FEATURE_MAP}']['f1_score']:.3f}, "
                        f"Acc={results_dict[f'final_fusion_{FEATURE_MAP}']['accuracy']:.3f}")



    
    print(f"Traditional feature predictions: {repr(results_dict)}")
    logger.info(f"Traditional feature predictions: {repr(results_dict)}")