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

from affinity import compute_affinities


def get_masked_token_predictions_batched(
        texts: Union[str, List[str]],
        topk: int = 5,
        candidate: str = None,
        mask_str: str = "[MASK]",
        batch_size: int = 16,
        device: str = None,
        causallm: bool = False,
        model_name: str = "roberta-base",
        revision: str = "",
        standard_bert = False,
        pll_eval = True
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

    print("batch size:", batch_size)

    if isinstance(texts, str):
        texts = [texts]

    if causallm or pll_eval:
        if pll_eval:
            print("IN PLL EVAL MODE")
            #Load a MLM if PLL
            if not revision:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForMaskedLM.from_pretrained(model_name)
            if revision:
                tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
                model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        else:
            #Else load a causal LM if not PLL
            if not revision:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
            if revision:
                tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
                model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        dev = next(model.parameters()).device
        print(f"Model is on device: {dev}")

        # use minicons Scorer on device
        if pll_eval:
            lm_scorer = scorer.MaskedLMScorer(model, tokenizer=tokenizer, device=device)
        else:
            lm_scorer = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)

        outputs_all = []
        batched_outputs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]

            prefixes = []
            queries = []

            for t in batch_texts:
                sent1, sent2 = t.split(".")[0] + ".", t.split(".")[1] + "."
                prefixes.append(sent1)  # append twice
                prefixes.append(sent1)
                q1 = sent2.split("<MASK>")[0].strip() + " easier" + sent2.split("<MASK>")[1]
                q2 = sent2.split("<MASK>")[0].strip() + " harder" + sent2.split("<MASK>")[1]
                queries.append(q1)
                queries.append(q2)

            # compute conditional scores
            scores = lm_scorer.conditional_score(prefixes, queries)

            batched_outputs.append((prefixes, scores))



        # process batched outputs
        for prefs, scores in batched_outputs:
            all_stuff = list(zip(prefs, scores))
            result_list = list(zip(all_stuff[::2], all_stuff[1::2]))
            # print(result_list)
            for (pref, score_easier), (pref, score_harder) in result_list:
                outputs_all.append((pref, score_easier, score_harder, "_", "_", "_", "_"))
        return outputs_all

    else:
        # MLM mode
        if not revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        dev = next(model.parameters()).device
        print(f"Model is on device: {dev}")

        outputs_all = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [re.sub("<MASK>", mask_str, t) for t in batch_texts]
            first_sents = [s.split(".")[0] + "." for s in batch_texts]
            cxn_list = []

            for s in first_sents:
                if "let alone" in s:
                    cxn_list.append("letalone")
                elif "much less" in s:
                    cxn_list.append("muchless")
                elif "not to mention" in s:
                    cxn_list.append("nottomention")
                elif "never mind" in s:
                    cxn_list.append("nevermind")
                elif "or" in s:
                    cxn_list.append("or")
                else:
                    raise ValueError("Could not identify connective in sentence for affinity computation.")


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
            # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(" easier, harder")))

            let_global, let_local, alone_global, alone_local = compute_affinities(
                batch_texts,
                model,
                tokenizer,
                device=device,
                standard_bert=standard_bert,
                cxn_list=cxn_list,
            )
            #print(let_global)

            for b_idx, text in enumerate(batch_texts):
                mask_positions = (input_ids[b_idx] == mask_token_id).nonzero(as_tuple=False).squeeze(1)
                results = {}

                for pos in mask_positions.tolist():
                    logits_for_mask = logits[b_idx, pos]
                    probs = torch.softmax(logits_for_mask, dim=-1)

                    # top-k predictions -- not currently used
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
                    # handle the fact that roberta uses Ġ for word-initial tokens and bert does not
                    # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(" easier, harder")))
                    if not standard_bert:
                        ez = "Ġeasier"
                        hd = "Ġharder"
                    else:
                        ez = "easier"
                        hd = "harder"
                    cand_easier = tokenizer.convert_tokens_to_ids(ez)
                    cand_harder = tokenizer.convert_tokens_to_ids(hd)
                    cand_easier_prob = probs[cand_easier].item()
                    cand_harder_prob = probs[cand_harder].item()

                    results[pos] = entry

                outputs_all.append((results, cand_easier_prob, cand_harder_prob, let_global[b_idx], let_local[b_idx], alone_global[b_idx], alone_local[b_idx]))

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
    parser.add_argument("--standard_bert", action="store_true", help="Use standard BERT model initial token format")
    parser.add_argument("--pll_eval", action="store_true", help="Use PLL evaluation mode with minicons (uses masked LM scorer)")
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
        revision=args.revision,
        standard_bert=args.standard_bert,
        pll_eval=args.pll_eval
    )

    #Torch memory summary
    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Prepare output DataFrame

    for r, (res, easier, harder, w1_g, w1_l, w2_g, w2_l) in zip(df.index, predictions):
        df.loc[r, "easier"] = easier
        df.loc[r, "harder"] = harder
        df.loc[r, "easier-harder"] = easier - harder
        df.loc[r, "Word1_Global_Affinity"] = w1_g
        df.loc[r, "Word1_Local_Affinity"] = w1_l
        df.loc[r, "Word2_Global_Affinity"] = w2_g
        df.loc[r, "Word2_Local_Affinity"] = w2_l

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