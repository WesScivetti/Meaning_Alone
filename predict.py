import os
import re
import torch
import pandas as pd
from typing import List, Union, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from minicons import scorer
from argparse import ArgumentParser


def get_masked_token_predictions_batched(
        texts: Union[str, List[str]],
        topk: int = 5,
        candidate: str = None,
        mask_str: str = "[MASK]",
        batch_size: int = 8,
        device: str = None,
        causallm: bool = False,
        model_name: str = "roberta-base",
        revision: str = ""
):
    """
    Compute predictions for masked tokens (MLM) or conditional log-probs (CLM).

    Args:
        texts: str or list of str. Sentences with <MASK>.
        topk: number of top-k predictions to return (MLM only).
        candidate: optional candidate token to score (MLM only).
        mask_str: mask string (default [MASK]).
        batch_size: batch size for MLM.
        device: "cuda" or "cpu". Defaults to auto-detect.
        causallm: if True, use causal LM mode with minicons.
        model_name: name of Hugging Face model to load.

    Returns:
        - If MLM: list of (results, prob_easier, prob_harder)
        - If CLM: list of dicts with conditional log-probs for easier/harder
    """

    if isinstance(texts, str):
        texts = [texts]

    if causallm:
        # use minicons Scorer on device
        lm_scorer = scorer.IncrementalLMScorer(model_name, device)

        outputs_all = []
        batched_outputs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]

            prefixes = []
            queries = []
            # print(batch_texts)
            # print("###")
            for t in batch_texts:
                # print(t)
                sent1, sent2 = t.split(".")[0] + ".", t.split(".")[1] + "."
                # print(sent1, sent2)
                # print("####")
                prefixes.append(sent1)  # append twice
                prefixes.append(sent1)
                q1 = sent2.split("<MASK>")[0].strip() + " easier" + sent2.split("<MASK>")[1]
                q2 = sent2.split("<MASK>")[0].strip() + " harder" + sent2.split("<MASK>")[1]
                # print(q1)
                # print(q2)
                # print("#####")
                queries.append(q1)
                queries.append(q2)

            # candidates

            # queries = [sent2.split("<MASK>")[0].strip() + " easier" + sent2.split("<MASK>")[1], sent2.split("<MASK>")[0].strip() + " harder" + sent2.split("<MASK>")[1]] * batch_size

            # print(len(prefixes), len(queries))
            # print(prefixes)
            # print(queries)

            # compute conditional scores
            scores = lm_scorer.conditional_score(prefixes, queries)

            batched_outputs.append((prefixes, scores))

            # for i in range(len(prefixes)):
            #   print(prefixes[i])
            #   print(queries[i])
            #   print(scores[i])

            # asfsamf

            # print(scores)

            # package result
        # print(batched_outputs)
        for prefs, scores in batched_outputs:
            all_stuff = list(zip(prefs, scores))
            result_list = list(zip(all_stuff[::2], all_stuff[1::2]))
            # print(result_list)
            for (pref, score_easier), (pref, score_harder) in result_list:
                outputs_all.append((pref, score_easier, score_harder))
        return outputs_all

    else:

        # print("HELLO")
        # MLM mode (like original)
        if not revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        outputs_all = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [re.sub("<MASK>", mask_str, t) for t in batch_texts]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            input_ids = enc["input_ids"]
            mask_token_id = tokenizer.mask_token_id

            with torch.no_grad():
                outputs = model(**enc)
            logits = outputs.logits

            for b_idx, text in enumerate(batch_texts):
                mask_positions = (input_ids[b_idx] == mask_token_id).nonzero(as_tuple=False).squeeze(1)
                results = {}

                for pos in mask_positions.tolist():
                    logits_for_mask = logits[b_idx, pos]
                    probs = torch.softmax(logits_for_mask, dim=-1)

                    # top-k predictions
                    top_probs, top_ids = torch.topk(probs, k=topk)
                    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
                    top = list(zip(top_tokens, top_probs.tolist()))

                    entry = {"top": top}

                    if candidate is not None:
                        cand_id = tokenizer.convert_tokens_to_ids(candidate)
                        cand_prob = probs[cand_id].item()
                        cand_log_prob = torch.log_softmax(logits_for_mask, dim=-1)[cand_id].item()
                        entry["candidate"] = (cand_prob, cand_log_prob)

                    # "easier" and "harder"
                    cand_easier = tokenizer.convert_tokens_to_ids("Ġeasier")
                    cand_harder = tokenizer.convert_tokens_to_ids("Ġharder")
                    cand_easier_prob = probs[cand_easier].item()
                    cand_harder_prob = probs[cand_harder].item()

                    results[pos] = entry

                outputs_all.append((results, cand_easier_prob, cand_harder_prob))

        return outputs_all

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_tsv", type=str, required=True, help="Input TSV file with 'sentence' column")
    parser.add_argument("--output_tsv", type=str, required=True, help="Output TSV file to save predictions")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--topk", type=int, default=2, help="Top-k predictions to return")
    parser.add_argument("--candidate", type=str, default=None, help="Optional candidate token to score")
    parser.add_argument("--mask_str", type=str, default="[MASK]", help="Mask string in the input sentences")
    parser.add_argument("--causallm", action="store_true", help="Use causal LM mode with minicons")
    parser.add_argument("--revision", type=str, default="", help="Model revision (optional)")
    args = parser.parse_args()

    # Load input CSV
    df = pd.read_csv(args.input_tsv,sep="\t")
    if 'Sentence' not in df.columns:
        raise ValueError("Input CSV must contain a 'Sentence' column")

    texts = df['Sentence'].tolist()

    # Get predictions
    predictions = get_masked_token_predictions_batched(
        texts,
        topk=args.topk,
        candidate=args.candidate,
        mask_str=args.mask_str,
        batch_size=args.batch_size,
        causallm=args.causallm,
        model_name=args.model_name,
        revision=args.revision
    )

    # Prepare output DataFrame

    for r, (res, easier, harder) in zip(df.index, predictions):
        df.loc[r, "easier"] = easier
        df.loc[r, "harder"] = harder
        df.loc[r, "easier-harder"] = easier - harder

    print(args)
    print(args.revision)
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