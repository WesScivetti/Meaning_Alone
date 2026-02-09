import pandas as pd
from argparse import ArgumentParser
import numpy as np
import pickle


def average_affinity_scores(input_file: str, output_file: str):
    """
    Averages global affinity scores for each connection from the input TSV file and writes to output TSV file.

    Args:
        input_file: Path to input TSV file with columns 'cxn_id' and 'affinity_score'.
        output_file: Path to output TSV file to write averaged scores.
    """
    # Read the input TSV file
    df = pd.read_csv(input_file, sep='\t')
    print(df.head())
    print(df.columns)

    mean_w1_global = df.groupby('Cxn')['Word1_Global_Affinity'].mean().reset_index()
    mean_w2_global = df.groupby('Cxn')['Word2_Global_Affinity'].mean().reset_index()
    #print(mean_w1_global["Cxn"])

    print(mean_w1_global)
    print(mean_w2_global)

    #print score for letalone:

    return mean_w1_global, mean_w2_global


def mean_one_checkpoint(input_file: str):
    """Computes the mean global affinity scores for each cxn from a single checkpoint's output file.
    Args:
        input_file: path to the tsv file containing the output for a single checkpoint.
    Returns:
        A dictionary with cxn as keys and their mean global affinity scores as values.
    """
    df = pd.read_csv(input_file, sep='\t')
    mean_w1_global = df.groupby('Cxn')['Word1_Global_Affinity'].mean().reset_index()
    mean_w2_global = df.groupby('Cxn')['Word2_Global_Affinity'].mean().reset_index()
    mean_affinities = {}
    mean_affs = []
    for cxn in ["letalone", "muchless", "nottomention", "nevermind"]:
        mean_aff = np.mean([mean_w1_global.loc[mean_w1_global['Cxn'] == cxn, 'Word1_Global_Affinity'].values[0],
                            mean_w2_global.loc[mean_w2_global['Cxn'] == cxn, 'Word2_Global_Affinity'].values[0]])
        mean_affinities[cxn] = mean_aff
        mean_affs.append(mean_aff)

    mean_affinities["average_pf"] = np.mean(mean_affs)
    return mean_affinities

def loop_through_checkpoints(models: list, revision_list: list):
    """Loops through all models and their checkpoints, computes mean global affinity scores for each cxn,
    and writes results to a tsv file.
    Args:
        models: list of model directory paths.
        revision_list: list of revision strings corresponding to checkpoints.
    """
    all_results = []
    for model_dir in models:
        model_name = model_dir.split("/")[-2]
        print(f"Processing model: {model_name}")
        for revision in revision_list:
            print("  Revision:", revision)
            input_file = f"{model_dir}/{revision}/combined_easy.tsv"
            mean_affinities = mean_one_checkpoint(input_file)
            for cxn, mean_aff in mean_affinities.items():
                all_results.append({
                    "model_name": model_name,
                    "revision": revision,
                    "cxn": cxn,
                    "mean_global_affinity": mean_aff
                })
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("outputs/mean_affinity_scores_across_checkpoints.tsv", sep="\t", index=False)
    print("Saved mean affinity scores across checkpoints to mean_affinity_scores_across_checkpoints.tsv")


def loop_through_models(model_directory_list: list):
    """loops through output files at the specified model directories, and computes their average global affinity for each cxn.
    Writes results into one big tsv.
    Args:
        model_directory_list: list of model directories containing output tsv files.
    """
    all_results = []
    for model_dir in model_directory_list:
        print(model_dir)
        input_file = f"{model_dir}/combined_easy.tsv"
        mean_w1_global, mean_w2_global = average_affinity_scores(input_file, None)
        mean_affs = []
        for cxn in ["letalone", "muchless", "nottomention", "nevermind"]:
            mean_aff = np.mean([mean_w1_global.loc[mean_w1_global['Cxn'] == cxn, 'Word1_Global_Affinity'].values[0],
                                mean_w2_global.loc[mean_w2_global['Cxn'] == cxn, 'Word2_Global_Affinity'].values[0]])
            mean_aff.append(mean_aff)
            all_results.append({
                "model_directory": model_dir,
                "cxn": cxn,
                "average_global_affinity": mean_aff
            })
        all_results.append({
            "model_directory": model_dir,
            "cxn": "average_pf",
            "average_global_affinity": np.mean(mean_aff)
        })
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("outputs/averaged_affinity_scores_across_models.tsv", sep="\t", index=False)
    print("Saved averaged affinity scores across models to averaged_affinity_scores_across_models.tsv")


def save_all_model_affinities(model_dir_list):
    """Saves all model affinities to a single pickle nested dictionary of the structre:
    {'model_name': {'cxn': [values]}}"""
    model_affinities = {}
    for model_dir in model_dir_list:
        input_file = f"{model_dir}/combined_easy.tsv"
        df = pd.read_csv(input_file, sep='\t')
        model_name = model_dir.split("/")[-3]

        #store in nested dictionary
        model_affinities[model_name] = {}
        for cxn in ["letalone", "muchless", "nottomention", "nevermind"]:
            w1_affs = df[df["Cxn"] == cxn]["Word1_Global_Affinity"].tolist()
            w2_affs = df[df["Cxn"] == cxn]["Word2_Global_Affinity"].tolist()
            affs = w1_affs + w2_affs
            model_affinities[model_name][cxn] = affs

    #save to pickle
    with open("outputs/all_model_affinities.pkl", "wb") as fout:
        pickle.dump(model_affinities, fout)



if __name__ == "__main__":
    #print("This module contains functions for averaging global affinity scores for each cxn from the output files in the outputs directory")
    MODEL_DIRECTORY_LIST = [
        "outputs/jhu-clsp/ettin-encoder-68m/NONE/",
        "outputs/jhu-clsp/ettin-encoder-150m/NONE/",
        "outputs/jhu-clsp/ettin-encoder-400m/NONE/",
        "outputs/jhu-clsp/ettin-encoder-1b/NONE/",
        "outputs/google/multiberts-seed_0/NONE/",
        "outputs/google/multiberts-seed_1/NONE/",
        "outputs/bert-base-uncased/NONE/",
        "outputs/bert-large-uncased/NONE/",
        "outputs/roberta-base/NONE/",
        "outputs/roberta-large/NONE/",
        "outputs/answerdotai/ModernBert-base/NONE/",
        "outputs/answerdotai/ModernBert-large/NONE/",
    ]
    all_revisions = [
        "step2999",
        "step5996",
        "step8994",
        "step11992",
        "step14991",
        "step17988",
        "step20986",
        "step23984",
        "step26982",
        "step29979",
        "step32976",
        "step35974",
        "step38972",
        "step41969",
        "step44967",
        "step47963",
        "step50960",
        "step53957",
        "step56955",
        "step59953",
        "step62950",
        "step65948",
        "step68944",
        "step71942",
        "step74940",
        "step77938",
        "step80935",
        "step83933",
        "step86931",
        "step89929",
        "step92926",
        "step95924",
        "step98921",
        "step101918",
        "step104916",
        "step107913",
        "step110911",
        "step113909",
        "step116906",
        "step119903",
        "step122900",
        "step125897",
        "step128895",
        "step131892",
        "step134889",
        "step137887",
        "step140884",
        "step143882",
        "step146880",
        "step149879",
    ]
    models = [
        "outputs/jhu-clsp/ettin-encoder-400m/",
        "outputs/jhu-clsp/ettin-encoder-1b/"
    ]
    loop_through_checkpoints(models, all_revisions)
    #loop_through_models(MODEL_DIRECTORY_LIST)
    #save_all_model_affinities(MODEL_DIRECTORY_LIST)
