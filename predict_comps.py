from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from minicons import scorer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os


def evaluate_comps_model(model_name: str, causallm: bool = False, revision: str = "", batch_size: int = 32):
    """
    evaluates the target model / revision on the comps dataset. For each model:
        Read good_bad sentences
        For each sentence pair, compute the log-probabilities of each sentence
        Create an output file BLiMP in the output directory
        File will contain columns: sentence good, sentence bad, phenomenon, log_prob_good, log_prob_bad
        If using a maskedLM, use psuedo log likelihood
    Args:
        model_name: name of Hugging Face model to load.
        causallm: if True, use causal LM mode with minicons.
        revision: optional revision of the model to load.
        batch_size: batch size to use for evaluation.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"



    if causallm:
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        dev = next(model.parameters()).device
        lm_scorer = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)

    else:
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(device)
        dev = next(model.parameters()).device
        lm_scorer = scorer.MaskedLMScorer(model, tokenizer=tokenizer, device=device)


    print(f"Model is on device: {dev}")
    print(lm_scorer)


    output_rows = []


    ds = load_dataset("kanishka/comps", "base")

    stimuli = []

    for row in ds["train"]:
        property1 = row["property"]
        sentence_good = row["prefix_acceptable"].capitalize() + " " + row["property_phrase"]
        sentence_bad = row["prefix_unacceptable"].capitalize() + " " + row["property_phrase"]
        stimuli.append([sentence_good, sentence_bad, property1])


    stimuli_dl = torch.utils.data.DataLoader(stimuli, batch_size=batch_size)

    # wrap in tqdm for progress bar
    for batch in tqdm(stimuli_dl, desc=f"Processing COMPS"):
        good, bad, property1 = batch

        good_scores = lm_scorer.sequence_score(good, reduction=lambda x: x.sum(0))
        bad_scores = lm_scorer.sequence_score(bad, reduction=lambda x: x.sum(0))

        for g_sent, b_sent, phen, g_score, b_score in zip(good, bad, property1, good_scores, bad_scores):
            output_rows.append({
                "sentence_good": g_sent,
                "sentence_bad": b_sent,
                "property": property1,
                "log_prob_good": g_score.item(),
                "log_prob_bad": b_score.item()
            })



    output_df = pd.DataFrame(output_rows)

    return output_df

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    argparse.add_argument("--output_dir",type=str, required=False, help="Path to output directory", default="outputs")
    argparse.add_argument("--causallm", action="store_true", help="Whether to use causal LM mode")
    argparse.add_argument("--revision", type=str, default="", help="Optional model revision to load")
    argparse.add_argument("--batch_size", type=int, default=64, help="Batch size to use")
    args = argparse.parse_args()

    output_df = evaluate_comps_model(
        model_name=args.model_name,
        causallm=args.causallm,
        revision=args.revision,
        batch_size=args.batch_size
    )

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    print("Done")

    #create full output path with outputs/model_name/revision
    if args.revision:
        revision_name = args.revision
    else:
        revision_name = "NONE"
    output_path = f"{args.output_dir}/{args.model_name}/{revision_name}/"

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "comps_base.tsv")
    output_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved results to {output_file}")