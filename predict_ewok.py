from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from minicons import scorer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os


def evaluate_ewok_model(model_name: str, causallm: bool = False, revision: str = "", batch_size: int = 32):
    """
    evaluates the target model / revision on the ewok dataset.
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


    ds = load_dataset("ewok-core/ewok-core-1.0")

    stimuli = []

    for row in ds["test"]:

        domain = row["Domain"]

        sentence_good1 = row["Context1"] + " " + row["Target1"]
        sentence_bad1 = row["Context1"] + " " + row["Target2"]
        sentence_good2 = row["Context2"] + " " + row["Target2"]
        sentence_bad2 = row["Context2"] + " " + row["Target1"]

        context1 = row["Context1"]
        context2 = row["Context2"]
        target1 = row["Target1"]
        target2 = row["Target2"]



        stimuli.append([context1, context2, target1, target2, domain])


    stimuli_dl = torch.utils.data.DataLoader(stimuli, batch_size=batch_size, pin_memory=True)


    # wrap in tqdm for progress bar
    for batch in tqdm(stimuli_dl, desc=f"Processing EWOK"):
        c1, c2, t1, t2, domain = batch

        prefixes_t1_c1 = c1
        prefixes_t1_c2 = c2
        prefixes_t2_c1 = c1
        prefixes_t2_c2 = c2

        targets_t1_c1 = t1
        targets_t1_c2 = t1
        targets_t2_c1 = t2
        targets_t2_c2 = t2

        scores_t1_c1 = lm_scorer.conditional_score(prefixes_t1_c1, targets_t1_c1)
        scores_t1_c2 = lm_scorer.conditional_score(prefixes_t1_c2, targets_t1_c2)
        scores_t2_c1 = lm_scorer.conditional_score(prefixes_t2_c1, targets_t2_c1)
        scores_t2_c2 = lm_scorer.conditional_score(prefixes_t2_c2, targets_t2_c2)


        for con1, con2, tar1, tar2, dom, s_t1_c1, s_t1_c2, s_t2_c1, s_t2_c2  in zip(c1, c2, t1, t2, domain, scores_t1_c1, scores_t1_c2, scores_t2_c1, scores_t2_c2):
            output_rows.append({
                "context1": con1,
                "context2": con2,
                "target1": tar1,
                "target2": tar2,
                "domain": dom,
                "log_prob_t1_giv_c1": s_t1_c1,
                "log_prob_t1_giv_c2": s_t1_c2,
                "log_prob_t2_giv_c1": s_t2_c1,
                "log_prob_t2_giv_c2": s_t2_c2
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

    output_df = evaluate_ewok_model(
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
    output_file = os.path.join(output_path, "ewok_core.tsv")
    output_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved results to {output_file}")