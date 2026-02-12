import os
import re
import torch.nn.functional as F
import torch
import pandas as pd
from typing import List, Union, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from minicons import scorer
from argparse import ArgumentParser
import huggingface_hub, transformers
from torch.utils.data import DataLoader


def get_log_probs_syntax(
        texts: Union[str, List[str]],
        batch_size: int = 1,
        device: str = None,
        model_name: str = "roberta-base",
        revision: str = "",
        causallm: bool = False,
):
    """
    Compute log probabilities for syntactic tests using masked LM or causal LM.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"


    if causallm:
        if not revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)

        model.to(device)
        lm_scorer = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)

    else:
        if not revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)

        model.to(device)
        lm_scorer = scorer.MaskedLMScorer(model, tokenizer=tokenizer, device=device)

    stimuli_dl = DataLoader(texts, batch_size=batch_size)

    scores = []
    if causallm:
        for batch in tqdm(stimuli_dl):
            good = batch
            scores_b = lm_scorer.sequence_score(good, reduction=lambda x: x.mean(0))
            scores_b = [float(score) for score in scores_b]
            scores.extend(scores_b)

    else:
        for batch in tqdm(stimuli_dl):
            good = batch
            scores_b = lm_scorer.sequence_score(good, reduction=lambda x: x.mean(0), PLL_metric='original')
            scores_b = [float(score) for score in scores_b]
            scores.extend(scores_b)

    return scores

# def process_df(
#         df: pd.DataFrame,
#         batch_size: int = 16,
#         device: str = None,
#         model_name: str = "roberta-base",
#         revision: str = "",
#         causallm: bool = False,
# ):
#     """
#     Process a DataFrame with a 'Sentence' column to compute log probabilities and save results.
#     """
#     texts = df['Sentence'].tolist()
#     scores = get_log_probs_syntax(
#         texts,
#         batch_size=batch_size,
#         device=device,
#         model_name=model_name,
#         revision=revision,
#         causallm=causallm
#     )
#     # Add scores to DataFrame
#     df['Score'] = scores
#     return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Input TSV file with 'sentence' column")
    parser.add_argument("--output_tsv", type=str, required=True, help="Output TSV file to save predictions")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--causallm", action="store_true", help="Use causal LM mode with minicons")
    parser.add_argument("--revision", type=str, default="", help="Model revision (optional)")
    args = parser.parse_args()

    # Load input CSV
    df = pd.read_csv(args.input_tsv,sep="\t")
    if 'Sentence' not in df.columns:
        raise ValueError("Input CSV must contain a 'Sentence' column")

    texts = df['Sentence'].tolist()

    #run get_log_probs_syntax on the sentences
    scores = get_log_probs_syntax(
        texts,
        batch_size=args.batch_size,
        device=None,
        model_name=args.model_name,
        revision=args.revision,
        causallm=args.causallm
    )
    df["Score"] = scores


    #Torch memory summary
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Prepare output DataFrame
    if args.revision:
        revision_name = args.revision
    else:
        revision_name = "NONE"

    # Build output directory: outputs/{model_name}/{revision}/
    print(revision_name)
    output_dir = os.path.join("outputs", args.model_name, revision_name)
    os.makedirs(output_dir, exist_ok=True)

    # Extract filename only (ignores any path user may pass)
    output_filename = os.path.basename(args.output_tsv)

    # Full save path
    output_path = os.path.join(output_dir, output_filename)

    # Save file
    df.to_csv(output_path, sep="\t", index=False)
    print(f"results saved to: {output_path}")